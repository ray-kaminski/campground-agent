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

        # Extract grounding chunks (place data)
        if hasattr(gm, 'grounding_chunks') and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                if hasattr(chunk, 'web') and chunk.web:
                    grounding_chunks.append({
                        "title": getattr(chunk.web, 'title', ''),
                        "uri": getattr(chunk.web, 'uri', ''),
                    })
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

    # If we have a widget token, inject it into the response
    if widget_token and hasattr(llm_response, 'content') and llm_response.content:
        content = llm_response.content

        # Find the text part and append the widget metadata
        if hasattr(content, 'parts') and content.parts:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    # Create the widget metadata block
                    widget_metadata = json.dumps({
                        "widget_token": widget_token,
                        "grounding_chunks": grounding_chunks
                    })

                    # Append to the response text
                    original_text = part.text
                    part.text = f"{original_text}\n\n[MAPS_WIDGET_DATA:{widget_metadata}]"
                    logger.info("Injected widget token into response")
                    break
    else:
        logger.info(f"No widget token found or no content. widget_token={widget_token is not None}")

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


# =============================================================================
# SUB-AGENTS
# =============================================================================

# MapsAgent: Handles external place queries using Google Maps grounding
# This agent is isolated because google_maps grounding cannot be combined with function tools
maps_agent = Agent(
    name="MapsAgent",
    description=f"""Use this agent for location and maps queries:

    1. FIND PLACES near the campground:
       - "Find restaurants nearby"
       - "What grocery stores are close?"
       - "Show me hiking trails"
       - "Where can I get gas?"

    2. GET DIRECTIONS to/from the campground:
       - "How do I get to the campground from [location]?"
       - "Directions to [place] from here"
       - "What's the best route to [destination]?"
       - "How far is [place] from the campground?"

    3. ROUTE INFORMATION:
       - "What roads should I take?"
       - "Are there any steep grades on the way?"
       - "Is the route RV-friendly?"

    The MapsAgent uses Google Maps for real-time place and routing information
    centered on {CAMPGROUND_NAME} in {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}.

    Location: {CAMPGROUND_LAT}, {CAMPGROUND_LNG}
    """,
    model="gemini-2.5-flash",
    instruction=f"""You help find places and directions near {CAMPGROUND_NAME} campground.
Location: {CAMPGROUND_LAT}, {CAMPGROUND_LNG}

Use Google Maps to answer questions about:
- Finding nearby places (restaurants, stores, trails, gas stations)
- Directions to/from the campground
- Distance and route information

RESPONSE FORMAT - ALWAYS RESPOND WITH VALID JSON:

For PLACES queries, return JSON like this:
```json
{{
  "type": "places",
  "summary": "Brief 1-sentence summary of what was found",
  "places": [
    {{
      "name": "Place Name",
      "address": "Full address",
      "rating": 4.5,
      "reviews": 123,
      "distance": "2.3 km",
      "drive_time": "5 min",
      "description": "Brief description of the place"
    }}
  ]
}}
```

For DIRECTIONS queries, return JSON like this:
```json
{{
  "type": "directions",
  "summary": "Brief summary of the route",
  "origin": "Starting location",
  "destination": "{CAMPGROUND_NAME}",
  "distance": "Total distance",
  "duration": "Total drive time",
  "steps": ["Step 1 description", "Step 2 description"]
}}
```

IMPORTANT:
- Always return valid JSON only, no other text
- Include 3-5 most relevant places
- Use the exact field names shown above""",
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
# ROOT AGENT (Orchestrator)
# =============================================================================

root_agent = Agent(
    name="CampgroundAssistant",
    model="gemini-2.5-flash",
    instruction=f"""You are a helpful assistant for campers staying at {CAMPGROUND_NAME}
in {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}.

You have access to three specialized tools:

1. **get_campground_info**: Use this ONLY for questions about what THIS campground offers:
   - Amenities at the campground (pool, wifi, laundry, etc.)
   - Site types available (RV, tent, hookups)
   - On-site activities AT the campground
   - Campground policies and rules
   - Contact information
   - Pricing

2. **MapsAgent**: Use this for ANY question about places OUTSIDE the campground:
   - Find places nearby (restaurants, gas stations, trails, stores, parks, attractions)
   - "Are there trails nearby?" → MapsAgent (NOT get_campground_info)
   - "Where can I find..." → MapsAgent
   - Get directions to/from anywhere
   - Route planning and distance questions
   - "How do I get to..." or "Directions to..."

3. **SearchAgent**: Use this for general questions needing current web info:
   - Weather forecasts
   - Local events
   - Regulations and permits
   - General information (history, tips, etc.)

ROUTING RULES - FOLLOW THESE EXACTLY:
1. Questions about EXTERNAL places (trails, restaurants, stores, attractions, parks nearby) → ALWAYS use MapsAgent
2. Questions about THIS campground's amenities, sites, rules, pricing → use get_campground_info
3. Questions needing current info (weather, events, news) → use SearchAgent
4. Directions or route questions → ALWAYS use MapsAgent

CRITICAL: When users ask about "trails nearby", "restaurants near here", "what's around", etc.,
these are EXTERNAL place queries and MUST go to MapsAgent for Google Maps grounded results.
Do NOT use get_campground_info for external places - it only has info about the campground itself.

HANDLING MAPSAGENT RESPONSES:
MapsAgent returns JSON data. When you receive JSON from MapsAgent:
- Pass the JSON through EXACTLY as received - do not modify or summarize it
- The frontend will parse and render it appropriately

RESPONSE FORMAT:
For campground_info responses, format naturally and add:
[RESPONSE_TYPE: campground_info]

For MapsAgent responses, pass through the JSON exactly as received.

For SearchAgent responses, format naturally and add:
[RESPONSE_TYPE: search] or [RESPONSE_TYPE: weather]

For greetings or general conversation:
[RESPONSE_TYPE: text]

Be friendly and helpful!
""",
    tools=[
        get_campground_info,
        AgentTool(agent=maps_agent),
        AgentTool(agent=search_agent),
    ],
)


# =============================================================================
# APP SETUP
# =============================================================================

app = App(root_agent=root_agent, name="app")
