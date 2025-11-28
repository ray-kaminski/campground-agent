# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Campground Assistant Agent using ADK Agent-as-Tool Pattern

Architecture:
- Root Agent: Orchestrates between sub-agents based on query type
- MapsAgent: Handles external place queries using google_maps grounding
- SearchAgent: Handles general web searches using google_search grounding
- CampgroundInfoTool: Function tool for campground-specific queries (grounded in canonical data)

The Agent-as-Tool pattern is used because google_maps and google_search grounding
cannot be combined with function declarations in the same request.
"""

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import google.auth
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps.app import App
from google.adk.models import LlmResponse
from google.adk.tools import AgentTool
from google.adk.tools import google_search
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from typing_extensions import override

if TYPE_CHECKING:
    from google.adk.models import LlmRequest

# Set up logging to see grounding metadata
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up environment for Vertex AI
_, project_id = google.auth.default()
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project_id)
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


# =============================================================================
# CAMPGROUND DATA LOADING
# =============================================================================

def load_campground_data() -> dict:
    """Load the canonical campground data for Bakersfield KOA."""
    # Try multiple possible paths for the canonical data
    possible_paths = [
        Path(__file__).parent.parent / "data" / "canonical.json",
        Path(__file__).parent.parent.parent / "lmtestbed" / "campgrounds" / "koa_bakersfield" / "canonical.json",
        Path("/Users/raymondkaminski/dev/lmtestbed/campgrounds/koa_bakersfield/canonical.json"),
    ]

    for path in possible_paths:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return data.get("campground", {})

    # Return default data if file not found
    return {
        "name": "Bakersfield KOA Journey",
        "location": {
            "city": "Bakersfield",
            "state": "CA",
            "latitude": 35.30871,
            "longitude": -119.039528
        }
    }


# Load campground data at module level
CAMPGROUND_DATA = load_campground_data()
CAMPGROUND_NAME = CAMPGROUND_DATA.get("name", "Bakersfield KOA Journey")
CAMPGROUND_LOCATION = CAMPGROUND_DATA.get("location", {})
CAMPGROUND_LAT = CAMPGROUND_LOCATION.get("latitude", 35.30871)
CAMPGROUND_LNG = CAMPGROUND_LOCATION.get("longitude", -119.039528)


# =============================================================================
# LOCATION-AWARE GOOGLE MAPS GROUNDING TOOL
# =============================================================================

class LocationAwareMapsGroundingTool(BaseTool):
    """Custom Google Maps grounding tool that includes lat/lng for location context.

    Unlike the default google_maps_grounding, this tool configures the retrieval_config
    with lat_lng to ground searches around the campground location.
    """

    def __init__(self, latitude: float, longitude: float):
        super().__init__(name='google_maps', description='google_maps')
        self.latitude = latitude
        self.longitude = longitude

    @override
    async def process_llm_request(
        self,
        *,
        tool_context: ToolContext,
        llm_request: "LlmRequest",
    ) -> None:
        llm_request.config = llm_request.config or types.GenerateContentConfig()
        llm_request.config.tools = llm_request.config.tools or []

        # Add Google Maps tool with widget enabled for interactive maps
        llm_request.config.tools.append(
            types.Tool(google_maps=types.GoogleMaps(enable_widget=True))
        )

        # Add tool_config with lat_lng for location grounding
        llm_request.config.tool_config = types.ToolConfig(
            retrieval_config=types.RetrievalConfig(
                lat_lng=types.LatLng(
                    latitude=self.latitude,
                    longitude=self.longitude
                )
            )
        )


# Create the location-aware maps tool for our campground
campground_maps_grounding = LocationAwareMapsGroundingTool(
    latitude=CAMPGROUND_LAT,
    longitude=CAMPGROUND_LNG
)


# =============================================================================
# CALLBACK TO CAPTURE GROUNDING METADATA AND INJECT WIDGET TOKEN
# =============================================================================

def after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse
) -> LlmResponse | None:
    """Callback to capture grounding metadata and store it in session state.

    This runs AFTER the model generates a response but BEFORE ADK processes it.
    We extract ALL rich grounding data and store it in session state via state_delta,
    which flows through the SSE stream to the frontend.
    
    Note: We use state_delta instead of custom_metadata because MapsAgent runs as
    a sub-agent via AgentTool, and custom_metadata doesn't propagate to root-level events.
    """
    widget_token = None
    places = []
    grounding_supports = []
    retrieval_queries = []

    # Extract grounding metadata
    gm = getattr(llm_response, 'grounding_metadata', None)
    
    # Also check candidates (some responses have it nested there)
    if not gm and hasattr(llm_response, 'candidates') and llm_response.candidates:
        for candidate in llm_response.candidates:
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                gm = candidate.grounding_metadata
                break

    if gm:
        # Extract widget token
        widget_token = getattr(gm, 'google_maps_widget_context_token', None)

        # Extract retrieval queries (what Maps searched for)
        retrieval_queries = getattr(gm, 'retrieval_queries', []) or []

        # Extract grounding chunks with FULL place data
        chunks = getattr(gm, 'grounding_chunks', []) or []
        for chunk in chunks:
            # Google Maps grounding
            if hasattr(chunk, 'maps') and chunk.maps:
                maps_chunk = chunk.maps
                place_data = {
                    "type": "maps",
                    "place_id": getattr(maps_chunk, 'place_id', None),
                    "title": getattr(maps_chunk, 'title', None),
                    "text": getattr(maps_chunk, 'text', None),
                    "uri": getattr(maps_chunk, 'uri', None),
                }
                
                # Extract review snippets if available
                place_answer_sources = getattr(maps_chunk, 'place_answer_sources', None)
                if place_answer_sources:
                    review_snippets = getattr(place_answer_sources, 'review_snippets', []) or []
                    place_data["reviews"] = []
                    for review in review_snippets:
                        place_data["reviews"].append({
                            "review": getattr(review, 'review', None),
                            "title": getattr(review, 'title', None),
                            "google_maps_uri": getattr(review, 'google_maps_uri', None),
                            "relative_time": getattr(review, 'relative_publish_time_description', None),
                        })
                
                if place_data.get("place_id"):
                    places.append(place_data)

            # Web grounding fallback
            elif hasattr(chunk, 'web') and chunk.web:
                web_chunk = chunk.web
                places.append({
                    "type": "web",
                    "title": getattr(web_chunk, 'title', None),
                    "uri": getattr(web_chunk, 'uri', None),
                    "domain": getattr(web_chunk, 'domain', None),
                })

        # Extract grounding supports (which text is grounded by which chunk)
        supports = getattr(gm, 'grounding_supports', []) or []
        for support in supports:
            segment = getattr(support, 'segment', None)
            grounding_supports.append({
                "text": getattr(segment, 'text', None) if segment else None,
                "start_index": getattr(segment, 'start_index', None) if segment else None,
                "end_index": getattr(segment, 'end_index', None) if segment else None,
                "chunk_indices": getattr(support, 'grounding_chunk_indices', []),
                "confidence_scores": getattr(support, 'confidence_scores', []),
            })

    # Store in session state if we have any grounding data
    # This flows through state_delta in SSE events
    if widget_token or places:
        callback_context.state["maps_grounding"] = {
            "widget_token": widget_token,
            "places": places,
            "place_ids": [p.get("place_id") for p in places if p.get("place_id")],
            "grounding_supports": grounding_supports,
            "retrieval_queries": retrieval_queries,
        }

    # Return None to let the response pass through unchanged
    return None


# =============================================================================
# CAMPGROUND INFO TOOL (Function-based, grounded in canonical data)
# =============================================================================

def get_campground_info(category: str) -> str:
    """Get information about this campground from our authoritative data.

    Use this tool when the user asks about THIS campground specifically:
    - Basic info (name, location, contact, hours)
    - Amenities (pool, showers, laundry, wifi, etc.)
    - Site types (RV sites, tent sites, hookups, lengths)
    - Activities (on-site and nearby)
    - Rules and policies (pets, quiet hours, check-in/out, cancellation)
    - Pricing (nightly rates, monthly rates, discounts)
    - Nearby attractions mentioned in our data

    Args:
        category: The category of information to retrieve. Options:
            - "basic_info": Name, address, contact, location
            - "amenities": All amenities and facilities
            - "site_types": RV sites, tent sites, cabins, hookups
            - "activities": On-site and nearby activities
            - "nearby_attractions": Attractions mentioned in our data
            - "rules": Policies, check-in/out, pet rules
            - "pricing": Rates and pricing information
            - "all": Complete campground information

    Returns:
        JSON string with the requested campground information.
    """
    data = CAMPGROUND_DATA

    if category == "basic_info":
        return json.dumps({
            "name": data.get("name"),
            "facility_id": data.get("facility_id"),
            "location": data.get("location"),
            "contact": data.get("contact"),
            "ratings": data.get("ratings"),
        }, indent=2)

    elif category == "amenities":
        return json.dumps({
            "amenities": data.get("amenities"),
            "facilities": data.get("descriptions", {}).get("facilities_and_infrastructure"),
        }, indent=2)

    elif category == "site_types":
        descriptions = data.get("descriptions", {})
        return json.dumps({
            "rv_sites": descriptions.get("facilities_and_infrastructure", {}).get("rv_sites"),
            "tent_sites": descriptions.get("facilities_and_infrastructure", {}).get("tent_sites"),
            "cabins_lodging": descriptions.get("facilities_and_infrastructure", {}).get("cabins_lodging"),
            "campsites": data.get("campsites"),
        }, indent=2)

    elif category == "activities":
        return json.dumps({
            "on_site": data.get("activities", {}).get("on_site"),
            "nearby": data.get("activities", {}).get("nearby"),
            "recreation": data.get("descriptions", {}).get("recreation_opportunities"),
        }, indent=2)

    elif category == "nearby_attractions":
        return json.dumps({
            "nearby_attractions": data.get("descriptions", {}).get("nearby_attractions"),
            "nearby_activities": data.get("activities", {}).get("nearby"),
        }, indent=2)

    elif category == "rules":
        return json.dumps({
            "policies": data.get("policies"),
            "check_in": data.get("check_in_out"),
        }, indent=2)

    elif category == "pricing":
        return json.dumps({
            "pricing": data.get("pricing"),
            "rates": data.get("rates"),
        }, indent=2)

    elif category == "all":
        return json.dumps(data, indent=2)

    else:
        return json.dumps({"error": f"Unknown category: {category}. Use one of: basic_info, amenities, site_types, activities, nearby_attractions, rules, pricing, all"})


def render_navigation_card(
    destination: str,
    origin: str = "",
    travel_mode: str = "DRIVING"
) -> str:
    """
    Render a navigation card with directions to a destination.
    
    USE THIS TOOL when the user asks for directions, routes, or how to get somewhere.
    
    Trigger phrases: "directions to", "how do I get to", "route to", "navigate to", "drive to"
    
    Examples:
    - "directions to Target" → destination="Target, Bakersfield, CA"
    - "how do I get to Starbucks" → destination="Starbucks, Bakersfield, CA"
    - "route to the nearest gas station" → destination="gas station, Bakersfield, CA"
    
    Args:
        destination: The destination name with city. Examples:
                    - "Target, Bakersfield, CA"
                    - "Starbucks, Bakersfield, CA"
                    - "Walmart, Bakersfield, CA"
                    Google Maps will find the nearest matching location.
        origin: The starting location. Leave empty/null to default to the campground.
               Only set if user explicitly mentions where they're starting from.
        travel_mode: DRIVING (default), WALKING, BICYCLING, or TRANSIT.
    
    Returns:
        JSON string with navigation data for the frontend to render.
    """
    # Default origin to campground if not specified
    if not origin or origin == "user_current_location":
        origin = f"{CAMPGROUND_NAME}, {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}"
    
    # Return structured data for the frontend to render
    navigation_data = {
        "type": "navigation",
        "action": "render_navigation_card",
        "origin": origin,
        "destination": destination,
        "travel_mode": travel_mode,
        "campground_location": {
            "lat": CAMPGROUND_LAT,
            "lng": CAMPGROUND_LNG,
            "name": CAMPGROUND_NAME
        }
    }
    return json.dumps(navigation_data)




# =============================================================================
# SUB-AGENTS
# =============================================================================

# MapsAgent: Handles SPECIFIC named place queries using Google Maps grounding
# Returns rich grounded data with photos and reviews
maps_agent = Agent(
    name="MapsAgent",
    description=f"""Use this agent when the user asks about a SPECIFIC NAMED place:

    ALWAYS USE MapsAgent FOR:
    - "Tell me about Olive Garden" → YES, named place
    - "What is Pizza Hut like?" → YES, named place
    - "Is there a Target nearby?" → YES, named place  
    - "Tell me about [restaurant/store name]" → YES, named place
    - "What are the hours for Walmart?" → YES, named place

    ALSO USE MapsAgent FOR generic discovery:
    - "Find restaurants nearby" → YES, use MapsAgent
    - "Where can I get gas?" → YES, use MapsAgent

    MapsAgent uses Google Maps grounding to provide RICH information:
    - Ratings and reviews
    - Hours of operation
    - Photos (rendered as interactive widget)
    - Detailed descriptions

    Location context: {CAMPGROUND_NAME} at {CAMPGROUND_LAT}, {CAMPGROUND_LNG}
    """,
    model="gemini-2.5-flash",
    instruction=f"""You are a helpful assistant for finding and describing places near {CAMPGROUND_NAME} campground.
Location: {CAMPGROUND_LAT}, {CAMPGROUND_LNG}

Handle TWO types of queries:

1. DISCOVERY queries ("find restaurants", "where can I get gas"):
   - List the top 3-5 nearby options
   - Include name, distance, rating, and a brief note about each
   - Mention which one you'd recommend and why

2. SPECIFIC place queries ("tell me about Olive Garden"):
   - Provide detailed info about that specific place
   - Include rating, hours, notable features, distance
   - Share camper-relevant tips

IMPORTANT: 
- Respond in natural, conversational English - NOT JSON
- The Google Maps grounding will automatically attach rich data (photos, reviews, map)
- Your text response will appear alongside the grounded maps widget

Example for "find restaurants nearby":
"There are several great dining options near the campground! The closest is Denny's (0.5 miles, 3.8★) 
which is open 24/7 - perfect for late arrivals. For Italian, Olive Garden (3 miles, 4.2★) has 
unlimited breadsticks and great family portions. If you want something quick, In-N-Out (2 miles, 4.5★) 
is a California classic. I'd recommend Olive Garden for a sit-down meal or In-N-Out for a quick bite."

Example for "tell me about Olive Garden":
"The Olive Garden on Rosedale Highway is about 3 miles from the campground, a quick 7-minute drive. 
It's an Italian-American chain with a 4.2 rating. They're famous for unlimited breadsticks. 
Open daily 11 AM - 10 PM. Great for families with generous portions."

This enables the rich Google Maps card with photos and reviews to render.""",
    tools=[campground_maps_grounding],
    after_model_callback=after_model_callback,
)


# SearchAgent: Handles general web searches using Google Search grounding
# This agent is isolated because google_search grounding works best alone
search_agent = Agent(
    name="SearchAgent",
    description=f"""Use this agent for general questions that need current web information.

    Examples of when to use SearchAgent:
    - "What's the weather forecast for this weekend?"
    - "Are there any local events happening?"
    - "What are the road conditions to Sequoia?"
    - "Tell me about the history of Bakersfield"
    - "What are the fishing regulations in Kern County?"

    The SearchAgent uses Google Search to find current, accurate information
    relevant to campers at {CAMPGROUND_NAME}.
    """,
    model="gemini-2.5-flash",
    instruction=f"""You are a research assistant for campers staying at {CAMPGROUND_NAME}
in {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}.

Your job is to answer questions that need current web information, such as:
- Weather forecasts
- Local events and happenings
- Road conditions
- Regulations and permits
- Historical or cultural information about the area
- Current prices or hours for attractions

Use Google Search to find accurate, up-to-date information.

RESPONSE FORMAT:
At the END of your response, include a response type tag on its own line:

For weather queries:
[RESPONSE_TYPE: weather]

For events, activities, or general search results:
[RESPONSE_TYPE: search]

Keep responses concise and cite sources inline.""",
    tools=[google_search],
)


# =============================================================================
# TRAILS DISCOVERY TOOLS AND AGENT
# =============================================================================

def render_trails_discovery(
    trails: str,
    summary: str = ""
) -> str:
    """
    Render a trails discovery card showing multiple hiking trails nearby.
    
    USE THIS TOOL when user asks about trails, hikes, or outdoor activities nearby.
    
    Trigger phrases: "trails nearby", "hiking trails", "where can I hike", "outdoor activities",
                    "nature walks", "what trails", "good hikes"
    
    Args:
        trails: JSON string array of trail objects, each with:
                - name: Trail name
                - distance_from_camp: Distance from campground (e.g., "5.2 miles")
                - difficulty: Easy, Moderate, Hard
                - length: Trail length (e.g., "3.5 mile loop")
                - highlights: Brief description of key features
                - rating: Star rating if available
        summary: A brief narrative introduction about trails in the area.
    
    Returns:
        JSON string for frontend to render trail discovery cards.
    """
    discovery_data = {
        "type": "trails_discovery",
        "action": "render_trails_discovery",
        "summary": summary,
        "trails": trails,
        "campground_location": {
            "lat": CAMPGROUND_LAT,
            "lng": CAMPGROUND_LNG,
            "name": CAMPGROUND_NAME
        }
    }
    return json.dumps(discovery_data)


def render_trail_details(
    trail_name: str,
    description: str = "",
    difficulty: str = "",
    length: str = "",
    elevation_gain: str = "",
    trail_type: str = "",
    best_seasons: str = "",
    highlights: str = "",
    warnings: str = "",
    amenities: str = "",
    address: str = ""
) -> str:
    """
    Render detailed information about a specific trail.
    
    USE THIS TOOL when user asks for details about a specific trail.
    
    Trigger phrases: "tell me about [trail name]", "details on [trail]", 
                    "what's [trail name] like", "[trail name] trail"
    
    Args:
        trail_name: Name of the trail
        description: Detailed narrative description of the trail experience
        difficulty: Easy, Moderate, Hard, Expert
        length: Trail length (e.g., "4.2 miles round trip")
        elevation_gain: Elevation change (e.g., "850 ft")
        trail_type: Loop, Out-and-back, Point-to-point
        best_seasons: Best times to visit
        highlights: Key features and scenic points
        warnings: Safety notes, hazards, or restrictions
        amenities: Parking, restrooms, water, etc.
        address: Trailhead address for directions
    
    Returns:
        JSON string for frontend to render trail details page.
    """
    details_data = {
        "type": "trail_details",
        "action": "render_trail_details",
        "trail": {
            "name": trail_name,
            "description": description,
            "difficulty": difficulty,
            "length": length,
            "elevation_gain": elevation_gain,
            "trail_type": trail_type,
            "best_seasons": best_seasons,
            "highlights": highlights,
            "warnings": warnings,
            "amenities": amenities,
            "address": address
        },
        "campground_location": {
            "lat": CAMPGROUND_LAT,
            "lng": CAMPGROUND_LNG,
            "name": CAMPGROUND_NAME
        }
    }
    return json.dumps(details_data)


# TrailsAgent: Specialized agent for trail/hiking discovery
# Uses function tools to render trail cards
trails_agent = Agent(
    name="TrailsAgent",
    description=f"""Use this agent for trail and hiking related queries:
    
    USE TrailsAgent FOR:
    - "What trails are nearby?" - initial discovery
    - "Where can I go hiking?" - initial discovery
    - "Tell me about [trail name]" - specific trail details
    - "What's [trail name] like?" - specific trail details
    
    Location context: {CAMPGROUND_NAME} at {CAMPGROUND_LAT}, {CAMPGROUND_LNG}
    """,
    model="gemini-2.5-flash",
    instruction=f"""You are a trail expert for {CAMPGROUND_NAME} ({CAMPGROUND_LOCATION.get('city', 'Unknown')}, {CAMPGROUND_LOCATION.get('state', 'CA')}).
Location: {CAMPGROUND_LAT}, {CAMPGROUND_LNG}

## WHEN TO USE CARD TOOLS:
Use render_trails_discovery or render_trail_details ONLY for:
- Initial discovery queries: "what trails are nearby?", "where can I hike?"
- Specific trail info requests: "tell me about Hart Park trail"

## WHEN TO USE PLAIN TEXT:
For follow-up questions, just respond with text - NO card needed:
- "What's the best time of year to visit?"
- "Is it good for kids?"
- "How long does it take?"
- "Any tips?"
- Opinion questions, clarifications, comparisons

## EXAMPLES:

CARD (initial query): "What trails are nearby?"
→ Call render_trails_discovery(...)

CARD (specific trail): "Tell me about Wind Wolves"  
→ Call render_trail_details(...)

TEXT (follow-up): "What's the best season?"
→ Just respond: "Spring (March-May) is ideal for wildflowers..."

TEXT (follow-up): "Is it dog friendly?"
→ Just respond: "Yes, dogs are allowed on leash..."

Be conversational for follow-ups. Only use cards for initial discovery or when user explicitly asks for trail info again.
""",
    tools=[render_trails_discovery, render_trail_details],
)


# =============================================================================
# ROOT AGENT (Orchestrator)
# =============================================================================

root_agent = Agent(
    name="CampgroundAssistant",
    model="gemini-2.5-flash",
    instruction=f"""You are a helpful assistant for campers staying at {CAMPGROUND_NAME}
in {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}.

You have access to these specialized tools:

1. **get_campground_info**: Use for questions about THIS campground:
   - Amenities (pool, wifi, laundry)
   - Site types (RV, tent, hookups)
   - Campground policies, rules, pricing

2. **render_navigation_card**: Use for DIRECTIONS queries:
   - "How do I get to Starbucks?" → destination="Starbucks, Bakersfield, CA"
   - "Directions to Target" → destination="Target, Bakersfield, CA"
   DEFAULT: Origin is the campground unless specified.

3. **render_trails_discovery**: Use for trail DISCOVERY queries:
   - "What trails are nearby?" → render_trails_discovery
   - "Where can I go hiking?" → render_trails_discovery
   - "Find hiking trails" → render_trails_discovery
   Call with summary text and JSON array of trails.

4. **render_trail_details**: Use for SPECIFIC trail queries:
   - "Tell me about Hart Park" → render_trail_details
   - "What's Wind Wolves Preserve like?" → render_trail_details
   Call with trail details (name, description, difficulty, etc.).

4. **MapsAgent**: Use for PLACES (parks, restaurants, stores, gas stations, attractions):
   - "What parks are nearby?" → MapsAgent (parks are places!)
   - "Find restaurants nearby" → MapsAgent
   - "Tell me about Olive Garden" → MapsAgent
   - "Where can I get gas?" → MapsAgent
   - "State parks near here" → MapsAgent

5. **SearchAgent**: Use for general info:
   - Weather forecasts
   - Local events
   - Regulations

ROUTING RULES - CHECK IN THIS ORDER:

**STEP 1: DIRECTIONS keywords?**
"directions", "how do I get to", "route to", "navigate to"
→ Use render_navigation_card

**STEP 2: EXPLICIT TRAIL/HIKING keywords?**
ONLY use trails tools when user explicitly says: "trail", "trails", "hiking", "hike"
→ Use render_trails_discovery for discovery OR render_trail_details for specific trail
NOTE: "parks" is NOT a trail query - parks are PLACES!

**STEP 3: PLACES keywords?**
Parks, restaurants, stores, gas, coffee, food, shopping, attractions
→ Use MapsAgent
IMPORTANT: "parks nearby", "state parks", "national parks" → MapsAgent (parks are places!)

**STEP 4: CAMPGROUND keywords?**
Amenities, rules, sites, hookups, pool, wifi
→ Use get_campground_info

**STEP 5: General info**
Weather, events, regulations
→ Use SearchAgent

CRITICAL DISTINCTION:
- "What PARKS are nearby?" → MapsAgent (parks are places)
- "What TRAILS are nearby?" → render_trails_discovery (trails are hiking paths)
- "Does [park name] have trails?" → MapsAgent first, then can discuss trails

RESPONSE FORMAT:
[RESPONSE_TYPE: campground_info] for campground queries
[RESPONSE_TYPE: search] or [RESPONSE_TYPE: weather] for SearchAgent
[RESPONSE_TYPE: text] for general conversation

Be friendly and helpful!
""",
    tools=[
        render_trails_discovery,  # For trail discovery
        render_trail_details,  # For specific trail details
        AgentTool(agent=maps_agent),  # For places queries (grounded maps)
        render_navigation_card,  # For directions
        get_campground_info,
        AgentTool(agent=search_agent),
    ],
)


# =============================================================================
# APP SETUP
# =============================================================================

app = App(root_agent=root_agent, name="app")
