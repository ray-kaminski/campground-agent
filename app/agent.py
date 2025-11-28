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
    """Callback to capture grounding metadata and inject widget token into response.

    This runs AFTER the model generates a response but BEFORE ADK processes it.
    Since ADK strips grounding_metadata, we inject the widget token directly into
    the response content so the frontend can use it.
    """
    logger.info("=== AFTER_MODEL_CALLBACK FIRED ===")
    widget_token = None
    grounding_chunks = []

    # Log all attributes of llm_response to understand its structure
    logger.info(f"llm_response type: {type(llm_response)}")
    logger.info(f"llm_response attrs: {[a for a in dir(llm_response) if not a.startswith('_')]}")

    # Check for grounding_metadata attribute directly
    if hasattr(llm_response, 'grounding_metadata') and llm_response.grounding_metadata:
        logger.info("FOUND grounding_metadata on llm_response!")
        gm = llm_response.grounding_metadata
        logger.info(f"grounding_metadata type: {type(gm)}")
        logger.info(f"grounding_metadata attrs: {[a for a in dir(gm) if not a.startswith('_')]}")

        # Log all grounding metadata fields
        logger.info(f"google_maps_widget_context_token = {getattr(gm, 'google_maps_widget_context_token', 'NOT FOUND')}")
        logger.info(f"grounding_chunks = {getattr(gm, 'grounding_chunks', 'NOT FOUND')}")
        logger.info(f"retrieval_metadata = {getattr(gm, 'retrieval_metadata', 'NOT FOUND')}")

        # Extract widget token
        if hasattr(gm, 'google_maps_widget_context_token') and gm.google_maps_widget_context_token:
            widget_token = gm.google_maps_widget_context_token
            logger.info(f"Captured widget token: {widget_token[:50]}...")
        else:
            logger.info(f"Widget token is empty/None: {gm.google_maps_widget_context_token if hasattr(gm, 'google_maps_widget_context_token') else 'NO ATTR'}")

        # Extract grounding chunks (place data) - check MAPS format first
        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            for i, chunk in enumerate(gm.grounding_chunks):
                chunk_data = {}
                
                # Check for Google Maps grounding (has place_id)
                if hasattr(chunk, 'maps') and chunk.maps:
                    maps_chunk = chunk.maps
                    place_id = getattr(maps_chunk, 'place_id', None)
                    if place_id:
                        chunk_data = {
                            "place_id": place_id,
                            "title": getattr(maps_chunk, 'title', ''),
                        }
                        logger.info(f"Extracted Maps place_id: {place_id}")
                
                # Fallback to web grounding
                elif hasattr(chunk, 'web') and chunk.web:
                    uri = getattr(chunk.web, 'uri', '')
                    chunk_data = {
                        "title": getattr(chunk.web, 'title', ''),
                        "uri": uri,
                    }
                
                if chunk_data:
                    grounding_chunks.append(chunk_data)
            
            logger.info(f"Total grounding_chunks extracted: {len(grounding_chunks)}")
    else:
        logger.info("No grounding_metadata on llm_response directly")

    # Check if grounding_metadata is nested in candidates
    if hasattr(llm_response, 'candidates') and llm_response.candidates:
        logger.info(f"Found {len(llm_response.candidates)} candidates")
        for i, candidate in enumerate(llm_response.candidates):
            logger.info(f"Candidate {i} attrs: {[a for a in dir(candidate) if not a.startswith('_')]}")
            if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                logger.info(f"FOUND grounding_metadata on candidate {i}!")
                gm = candidate.grounding_metadata
                logger.info(f"candidate grounding_metadata: {gm}")
                if hasattr(gm, 'google_maps_widget_context_token') and gm.google_maps_widget_context_token:
                    widget_token = gm.google_maps_widget_context_token
                    logger.info(f"Captured widget token from candidate: {widget_token[:50]}...")

    # Inject widget data if we have token OR grounding_chunks (for fallback rendering)
    if (widget_token or grounding_chunks) and hasattr(llm_response, 'content') and llm_response.content:
        content = llm_response.content

        # Find the text part and append the widget metadata
        if hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    # Create the widget metadata block
                    widget_metadata = json.dumps({
                        "widget_token": widget_token,
                        "grounding_chunks": grounding_chunks,
                        "place_ids": [c.get("place_id") for c in grounding_chunks if c.get("place_id")]
                    })

                    # Append to the response text
                    original_text = part.text
                    part.text = f"{original_text}\n\n[MAPS_WIDGET_DATA:{widget_metadata}]"
                    logger.info(f"Injected widget data: token={'yes' if widget_token else 'no'}, places={len(grounding_chunks)}")
                    break
    else:
        logger.info(f"No widget data to inject. widget_token={widget_token is not None}, chunks={len(grounding_chunks)}")

    # Return None to let the (possibly modified) response pass through
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
    description=f"""Use this agent for ALL trail and hiking related queries:
    
    ALWAYS USE TrailsAgent FOR:
    - "What trails are nearby?"
    - "Where can I go hiking?"
    - "Find hiking trails"
    - "Good hikes near the campground"
    - "Nature walks nearby"
    - "Tell me about [trail name]"
    - "What's [trail name] like?"
    - "Outdoor activities nearby"
    
    Location context: {CAMPGROUND_NAME} at {CAMPGROUND_LAT}, {CAMPGROUND_LNG}
    """,
    model="gemini-2.5-flash",
    instruction=f"""⚠️ MANDATORY: You MUST call render_trails_discovery or render_trail_details. DO NOT respond with plain text.

You are a trail expert for {CAMPGROUND_NAME} ({CAMPGROUND_LOCATION.get('city', 'Unknown')}, {CAMPGROUND_LOCATION.get('state', 'CA')}).
Location: {CAMPGROUND_LAT}, {CAMPGROUND_LNG}

## DISCOVERY QUERIES ("trails nearby", "where to hike"):
CALL render_trails_discovery with summary and trails JSON array.

## SPECIFIC TRAIL ("tell me about X trail"):  
CALL render_trail_details with trail info.

Example - YOU MUST format calls exactly like this:

For discovery:
render_trails_discovery(summary="The area has great trails...", trails='[{{"name":"Trail A","distance_from_camp":"5 miles","difficulty":"Easy","length":"2 mile loop","highlights":"River views","rating":"4.5"}}]')

For details:
render_trail_details(trail_name="Trail A", description="A beautiful trail...", difficulty="Easy", length="2 miles", elevation_gain="100 ft", trail_type="Loop", best_seasons="Spring", highlights="River, birds", warnings="None", amenities="Parking", address="Trail A, City, State")

⚠️ NEVER respond with text. ALWAYS call one of these tools!
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

4. **MapsAgent**: Use for places (restaurants, stores, gas stations):
   - "Find restaurants nearby" → MapsAgent
   - "Tell me about Olive Garden" → MapsAgent
   - "Where can I get gas?" → MapsAgent

5. **SearchAgent**: Use for general info:
   - Weather forecasts
   - Local events
   - Regulations

ROUTING RULES - CHECK IN THIS ORDER:

**STEP 1: DIRECTIONS keywords?**
"directions", "how do I get to", "route to", "navigate to"
→ Use render_navigation_card

**STEP 2: TRAIL/HIKING keywords?**
"trail", "trails", "hiking", "hike", "nature walk", "outdoor activities"
→ Use render_trails_discovery for discovery OR render_trail_details for specific trail

**STEP 3: PLACES keywords?**
Restaurant, store, gas, coffee, food, shopping
→ Use MapsAgent

**STEP 4: CAMPGROUND keywords?**
Amenities, rules, sites, hookups, pool, wifi
→ Use get_campground_info

**STEP 5: General info**
Weather, events, regulations
→ Use SearchAgent

CRITICAL: Trail queries use render_trails_discovery or render_trail_details, NOT MapsAgent!
- "What trails are nearby?" → render_trails_discovery
- "Where can I hike?" → render_trails_discovery
- "Tell me about Hart Park" → render_trail_details

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
