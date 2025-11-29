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
- PlacesAgent: Uses Google Places API (function tools) for real place data
- TrailsAgent: Uses google_search grounding for trail narratives  
- SearchAgent: Uses google_search grounding for general web queries
- CampgroundInfoTool: Function tool for campground-specific queries
- render_navigation_card: Function tool for directions

The Agent-as-Tool pattern is used because grounding tools (google_search, google_maps)
cannot be combined with function declarations in the same request.

PlacesAgent returns structured data (distances, ratings, photos) that can be
passed to TrailsAgent for rich narratives combining real data with web research.
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import google.auth
import httpx

# =============================================================================
# PLACES CACHE - Cache by place_id for faster responses
# =============================================================================

PLACES_CACHE: Dict[str, dict] = {}  # {place_id: {data: ..., timestamp: ...}}
PLACES_CACHE_TTL = 3600  # 1 hour cache TTL

def get_cached_place(place_id: str) -> dict | None:
    """Get cached place data if not expired."""
    if place_id in PLACES_CACHE:
        cached = PLACES_CACHE[place_id]
        if time.time() - cached["timestamp"] < PLACES_CACHE_TTL:
            return cached["data"]
    return None

def cache_place(place_id: str, data: dict) -> None:
    """Cache place data with timestamp."""
    PLACES_CACHE[place_id] = {"data": data, "timestamp": time.time()}
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

# Google Places API key (same as used in server.py)
GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY", "AIzaSyB_m4hT4FHoxiZ_JcuHteaVpGbSmwxvmz4")

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


# Note: LocationAwareMapsGroundingTool available if needed for future maps features
# campground_maps_grounding = LocationAwareMapsGroundingTool(
#     latitude=CAMPGROUND_LAT,
#     longitude=CAMPGROUND_LNG
# )


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
            if hasattr(chunk, 'maps') and chunk.maps:
                maps_chunk = chunk.maps
                place_data = {
                    "type": "maps",
                    "place_id": getattr(maps_chunk, 'place_id', None),
                    "title": getattr(maps_chunk, 'title', None),
                    "text": getattr(maps_chunk, 'text', None),
                    "uri": getattr(maps_chunk, 'uri', None),
                }
                
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

            elif hasattr(chunk, 'web') and chunk.web:
                web_chunk = chunk.web
                places.append({
                    "type": "web",
                    "title": getattr(web_chunk, 'title', None),
                    "uri": getattr(web_chunk, 'uri', None),
                    "domain": getattr(web_chunk, 'domain', None),
                })

        # Extract grounding supports
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

def _calculate_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two points in miles using Haversine formula."""
    R = 3959  # Earth's radius in miles
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lng = math.radians(lng2 - lng1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lng/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def _estimate_drive_time(distance_miles: float) -> str:
    """Estimate drive time based on distance (assuming ~35 mph average with traffic)."""
    minutes = int(distance_miles / 35 * 60)
    if minutes < 60:
        return f"~{minutes} min drive"
    else:
        hours = minutes // 60
        mins = minutes % 60
        if mins > 0:
            return f"~{hours}h {mins}min drive"
        return f"~{hours}h drive"


def search_osm_trails(
    tool_context: ToolContext
) -> str:
    """
    Search OpenStreetMap for hiking trails near the campground.
    
    Returns trail-specific data: names, difficulty (sac_scale), surface type.
    OSM often has more trail data than Google Places.
    
    Returns:
        JSON array of trails from OpenStreetMap with difficulty and surface info.
    """
    cache_key = f"osm_trails_{CAMPGROUND_LAT}_{CAMPGROUND_LNG}"
    cached = get_cached_place(cache_key)
    if cached:
        return json.dumps(cached)
    
    try:
        # Overpass API query for hiking trails
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Query for official hiking trails per OSM US Trails guidance
        # https://openstreetmap.us/our-work/trails/how-to-map/
        # Focus on: highway=path with name, foot access, or operator
        query = f"""
        [out:json][timeout:25];
        (
          relation["route"="hiking"](around:80000,{CAMPGROUND_LAT},{CAMPGROUND_LNG});
          way["highway"="path"]["name"]["foot"](around:80000,{CAMPGROUND_LAT},{CAMPGROUND_LNG});
          way["highway"="path"]["name"]["operator"](around:80000,{CAMPGROUND_LAT},{CAMPGROUND_LNG});
          way["highway"="path"]["name"~"Trail|Preserve|Loop|Canyon",i](around:80000,{CAMPGROUND_LAT},{CAMPGROUND_LNG});
        );
        out body center;
        """
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(overpass_url, data={"data": query})
            
            if response.status_code != 200:
                return json.dumps({"error": f"Overpass API error: {response.status_code}", "trails": []})
            
            data = response.json()
            elements = data.get("elements", [])
            
            # Process and deduplicate trails
            trails_by_name = {}
            
            # Words that indicate it's a street, not a trail
            street_words = ["street", "drive", "avenue", "road", "boulevard", "lane", "court", "place", "way", "circle", "highway", "freeway"]
            # Words that indicate it's likely a trail
            trail_words = ["trail", "path", "preserve", "canyon", "peak", "loop", "hike", "wilderness", "nature"]
            
            for element in elements:
                tags = element.get("tags", {})
                name = tags.get("name")
                
                if not name:
                    continue
                
                name_lower = name.lower()
                
                # Skip if it looks like a street
                if any(word in name_lower for word in street_words):
                    continue
                
                # Also skip generic names and non-trails
                if name_lower in ["gym entrance", "park entrance", "parking", "restroom", "sidewalk"]:
                    continue
                
                # Skip informal/social trails (per OSM tagging guidelines)
                if tags.get("informal") == "yes":
                    continue
                
                # Skip if access is prohibited
                if tags.get("access") in ["no", "private"]:
                    continue
                
                # Get center coordinates
                if element.get("type") == "way":
                    center = element.get("center", {})
                    lat = center.get("lat", 0)
                    lng = center.get("lon", 0)
                elif element.get("type") == "node":
                    lat = element.get("lat", 0)
                    lng = element.get("lon", 0)
                elif element.get("type") == "relation":
                    # Relations don't have center by default, skip distance calc
                    lat, lng = 0, 0
                else:
                    lat, lng = 0, 0
                
                # Calculate distance if we have coordinates
                distance_miles = None
                if lat and lng:
                    distance_miles = round(_calculate_distance(CAMPGROUND_LAT, CAMPGROUND_LNG, lat, lng), 1)
                
                # SAC scale difficulty mapping
                sac_scale = tags.get("sac_scale", "")
                difficulty_map = {
                    "hiking": "Easy",
                    "mountain_hiking": "Moderate", 
                    "demanding_mountain_hiking": "Difficult",
                    "alpine_hiking": "Very Difficult",
                    "demanding_alpine_hiking": "Expert",
                    "difficult_alpine_hiking": "Expert"
                }
                difficulty = difficulty_map.get(sac_scale, tags.get("trail_visibility", "Unknown"))
                
                # Google Maps URL for the trail location
                maps_url = f"https://www.google.com/maps?q={lat},{lng}&z=15" if lat and lng else None
                
                trail_data = {
                    "name": name,
                    "osm_id": element.get("id"),
                    "type": element.get("type"),
                    "lat": lat,  # For Google Places lookup
                    "lng": lng,
                    "maps_url": maps_url,  # Opens Google Maps at this location
                    "distance_miles": distance_miles,
                    "difficulty": difficulty,
                    "surface": tags.get("surface", ""),
                    "sac_scale": sac_scale,
                    "operator": tags.get("operator", ""),
                    "foot_access": tags.get("foot", ""),
                    "description": tags.get("description", ""),
                    "length_km": tags.get("length"),
                    "source": "OpenStreetMap"
                }
                
                # Keep the one with most data or closest
                if name not in trails_by_name:
                    trails_by_name[name] = trail_data
                elif distance_miles and (trails_by_name[name].get("distance_miles") is None or 
                                         distance_miles < trails_by_name[name]["distance_miles"]):
                    trails_by_name[name] = trail_data
            
            # Convert to list and sort by distance
            trails = list(trails_by_name.values())
            trails.sort(key=lambda x: x["distance_miles"] if x["distance_miles"] else 999)
            
            # Limit to top 15
            trails = trails[:15]
            
            result = {
                "source": "OpenStreetMap",
                "trails_found": len(trails),
                "trails": trails
            }
            
            cache_place(cache_key, result)
            return json.dumps(result)
            
    except Exception as e:
        return json.dumps({"error": str(e), "trails": []})


def _enrich_osm_trail_with_google(trail: dict) -> dict:
    """Look up an OSM trail in Google Places using its coordinates."""
    lat = trail.get("lat", 0)
    lng = trail.get("lng", 0)
    name = trail.get("name", "")
    
    if not lat or not lng:
        return trail
    
    # Check cache first
    cache_key = f"osm_enriched_{lat}_{lng}"
    cached = get_cached_place(cache_key)
    if cached:
        trail.update(cached)
        return trail
    
    try:
        # Search Google Places near the trail's coordinates
        search_url = "https://places.googleapis.com/v1/places:searchNearby"
        headers = {
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.rating,places.userRatingCount,places.photos",
            "Content-Type": "application/json"
        }
        
        body = {
            "locationRestriction": {
                "circle": {
                    "center": {"latitude": lat, "longitude": lng},
                    "radius": 2000  # 2km radius around the trail
                }
            },
            "includedTypes": ["park", "hiking_area", "national_park", "campground"],
            "maxResultCount": 1
        }
        
        with httpx.Client(timeout=5.0) as client:
            response = client.post(search_url, headers=headers, json=body)
            
            if response.status_code == 200:
                data = response.json()
                places = data.get("places", [])
                
                if places:
                    place = places[0]
                    nearby_name = place.get("displayName", {}).get("text", "")
                    enrichment = {
                        "google_place_id": place.get("id"),
                        "google_name": nearby_name,
                        "rating": place.get("rating"),
                        "rating_inferred": True,  # Rating is from nearby place, not trail itself
                        "rating_source": nearby_name,  # Which place the rating came from
                        "review_count": place.get("userRatingCount"),
                    }
                    
                    # Get photo URL
                    photos = place.get("photos", [])
                    if photos:
                        photo_name = photos[0].get("name", "")
                        if photo_name:
                            enrichment["photo_url"] = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=400&key={GOOGLE_PLACES_API_KEY}"
                    
                    # Cache the enrichment
                    cache_place(cache_key, enrichment)
                    trail.update(enrichment)
    
    except Exception:
        pass  # Fail silently, just use OSM data
    
    return trail


def discover_all_trails(
    tool_context: ToolContext
) -> str:
    """
    Discover trails from both OSM and Google Places, with smart caching.
    
    This is the main trail discovery function that:
    1. Gets trails from OpenStreetMap (names, surface, coordinates)
    2. Gets places from Google Places (preserves, parks with ratings)
    3. Enriches top 5 OSM trails with Google Places data using coordinates
    4. Caches everything for fast subsequent queries
    
    Returns:
        JSON with combined trail data from both sources.
    """
    # Check for cached combined results
    cache_key = f"all_trails_{CAMPGROUND_LAT}_{CAMPGROUND_LNG}"
    cached = get_cached_place(cache_key)
    if cached:
        tool_context.state["discovered_trails"] = cached
        return json.dumps(cached)
    
    combined_trails = []
    
    # Step 1: Get OSM trails
    osm_result = json.loads(search_osm_trails(tool_context))
    osm_trails = osm_result.get("trails", [])
    
    # Step 2: Get Google Places
    google_result = json.loads(search_nearby_trails(tool_context))
    google_places = google_result.get("places", [])
    
    # Step 3: Enrich top 5 OSM trails with Google data
    for i, trail in enumerate(osm_trails[:5]):
        enriched = _enrich_osm_trail_with_google(trail)
        enriched["source"] = "OSM+Google" if enriched.get("rating") else "OSM"
        combined_trails.append(enriched)
    
    # Add remaining OSM trails (not enriched)
    for trail in osm_trails[5:10]:
        trail["source"] = "OSM"
        combined_trails.append(trail)
    
    # Step 4: Add Google Places that aren't duplicates
    osm_names = {t["name"].lower() for t in osm_trails}
    for place in google_places:
        if place["name"].lower() not in osm_names:
            place["source"] = "Google"
            combined_trails.append(place)
    
    # Sort by distance
    combined_trails.sort(key=lambda x: x.get("distance_miles") or 999)
    
    result = {
        "total_trails": len(combined_trails),
        "osm_count": len(osm_trails),
        "google_count": len(google_places),
        "enriched_count": sum(1 for t in combined_trails if t.get("source") == "OSM+Google"),
        "trails": combined_trails[:15]  # Top 15
    }
    
    # Cache the combined results
    cache_place(cache_key, result)
    tool_context.state["discovered_trails"] = result
    
    return json.dumps(result)


def search_nearby_trails(
    tool_context: ToolContext
) -> str:
    """
    Search for hiking trails, nature preserves, and outdoor recreation near the campground.
    
    Call this FIRST for trail discovery. Returns trails with distances, ratings, and photos.
    
    Returns:
        JSON array of nearby trails with real data from Google Places API.
    """
    # Check for cached nearby trails (use a special cache key)
    cache_key = f"nearby_trails_{CAMPGROUND_LAT}_{CAMPGROUND_LNG}"
    cached = get_cached_place(cache_key)
    if cached:
        tool_context.state["nearby_trails"] = cached["places"]
        return json.dumps(cached)
    
    try:
        # Search for outdoor places near the campground
        search_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.photos,places.types",
            "Content-Type": "application/json"
        }
        
        # Search for parks, trails, and nature preserves near the campground
        body = {
            "textQuery": "hiking trails parks nature preserves outdoor recreation",
            "maxResultCount": 8,
            "locationBias": {
                "circle": {
                    "center": {"latitude": CAMPGROUND_LAT, "longitude": CAMPGROUND_LNG},
                    "radius": 50000  # 50km radius (~30 miles) - max allowed
                }
            }
        }
        
        with httpx.Client(timeout=15.0) as client:
            response = client.post(search_url, headers=headers, json=body)
            
            if response.status_code != 200:
                return json.dumps({"error": f"Places API error: {response.status_code}", "places": []})
            
            data = response.json()
            places_list = data.get("places", [])
            
            results = []
            for place in places_list:
                place_id = place.get("id", "")
                location = place.get("location", {})
                place_lat = location.get("latitude", 0)
                place_lng = location.get("longitude", 0)
                
                distance_miles = _calculate_distance(CAMPGROUND_LAT, CAMPGROUND_LNG, place_lat, place_lng)
                drive_time = _estimate_drive_time(distance_miles)
                
                # Get first photo URL - ensure it's valid
                photo_url = None
                photos = place.get("photos", [])
                if photos:
                    photo_name = photos[0].get("name", "")
                    if photo_name:
                        photo_url = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
                
                place_data = {
                    "name": place.get("displayName", {}).get("text", "Unknown"),
                    "place_id": place_id,
                    "distance_miles": round(distance_miles, 1),
                    "drive_time": drive_time,
                    "rating": place.get("rating"),
                    "review_count": place.get("userRatingCount"),
                    "photo_url": photo_url,
                    "types": place.get("types", [])
                }
                
                results.append(place_data)
                
                # Cache individual place data
                if place_id:
                    cache_place(place_id, place_data)
            
            # Sort by distance
            results.sort(key=lambda x: x["distance_miles"])
            
            # Store in state
            tool_context.state["nearby_trails"] = results
            
            # Cache the full results
            result_data = {
                "campground": CAMPGROUND_NAME,
                "places_found": len(results),
                "places": results
            }
            cache_place(cache_key, result_data)
            
            return json.dumps({
                "campground": CAMPGROUND_NAME,
                "places_found": len(results),
                "places": results
            })
            
    except Exception as e:
        return json.dumps({"error": str(e), "places": []})


def get_place_details(
    place_name: str,
    tool_context: ToolContext
) -> str:
    """
    Get detailed information about a specific trail, park, or outdoor location.
    
    Use this to get:
    - Multiple photo URLs for embedding
    - Exact distance and drive time from campground
    - Google rating and review count
    - Address and website
    
    Args:
        place_name: Name of the place (e.g., "Wind Wolves Preserve", "Hart Memorial Park")
    
    Returns:
        JSON with detailed place data including multiple photo URLs.
    """
    try:
        search_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.photos,places.regularOpeningHours,places.websiteUri,places.editorialSummary",
            "Content-Type": "application/json"
        }
        
        body = {
            "textQuery": f"{place_name} California",
            "maxResultCount": 1,
            "locationBias": {
                "circle": {
                    "center": {"latitude": CAMPGROUND_LAT, "longitude": CAMPGROUND_LNG},
                    "radius": 50000  # max allowed by Places API
                }
            }
        }
        
        with httpx.Client(timeout=10.0) as client:
            response = client.post(search_url, headers=headers, json=body)
            
            if response.status_code != 200:
                return json.dumps({"error": f"API error: {response.status_code}", "place_name": place_name})
            
            data = response.json()
            places = data.get("places", [])
            
            if not places:
                return json.dumps({"error": "Place not found", "place_name": place_name})
            
            place = places[0]
            location = place.get("location", {})
            place_lat = location.get("latitude", 0)
            place_lng = location.get("longitude", 0)
            
            distance_miles = _calculate_distance(CAMPGROUND_LAT, CAMPGROUND_LNG, place_lat, place_lng)
            drive_time = _estimate_drive_time(distance_miles)
            
            # Build multiple photo URLs
            photo_urls = []
            photos = place.get("photos", [])[:5]
            for photo in photos:
                photo_name = photo.get("name", "")
                if photo_name:
                    photo_urls.append(f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}")
            
            display_name = place.get("displayName", {}).get("text", place_name)
            
            result = {
                "name": display_name,
                "place_id": place.get("id", ""),
                "address": place.get("formattedAddress", ""),
                "distance_miles": round(distance_miles, 1),
                "drive_time": drive_time,
                "rating": place.get("rating"),
                "review_count": place.get("userRatingCount"),
                "website": place.get("websiteUri"),
                "summary": place.get("editorialSummary", {}).get("text"),
                "photo_urls": photo_urls,
                "photo_count": len(photo_urls),
                # Ready-to-use markdown
                "photo_markdown": f"![{display_name}]({photo_urls[0]})" if photo_urls else None
            }
            
            # Store in state
            if "places_data" not in tool_context.state:
                tool_context.state["places_data"] = {}
            tool_context.state["places_data"][place_name] = result
            
            return json.dumps(result)
            
    except Exception as e:
        return json.dumps({"error": str(e), "place_name": place_name})


def search_nearby_places(
    query: str,
    tool_context: ToolContext
) -> str:
    """
    Search for any type of place near the campground (restaurants, parks, stores, etc.).
    
    Args:
        query: What to search for (e.g., "restaurants", "grocery stores", "gas stations")
    
    Returns:
        JSON array of nearby places with distances, ratings, and photos.
    """
    cache_key = f"nearby_{query.replace(' ', '_')}_{CAMPGROUND_LAT}_{CAMPGROUND_LNG}"
    cached = get_cached_place(cache_key)
    if cached:
        return json.dumps(cached)
    
    try:
        search_url = "https://places.googleapis.com/v1/places:searchText"
        headers = {
            "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
            "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.userRatingCount,places.photos,places.types",
            "Content-Type": "application/json"
        }
        
        body = {
            "textQuery": query,
            "maxResultCount": 10,
            "locationBias": {
                "circle": {
                    "center": {"latitude": CAMPGROUND_LAT, "longitude": CAMPGROUND_LNG},
                    "radius": 50000
                }
            }
        }
        
        with httpx.Client(timeout=15.0) as client:
            response = client.post(search_url, headers=headers, json=body)
            
            if response.status_code != 200:
                return json.dumps({"error": f"Places API error: {response.status_code}", "places": []})
            
            data = response.json()
            places_list = data.get("places", [])
            
            results = []
            for place in places_list:
                place_id = place.get("id", "")
                location = place.get("location", {})
                place_lat = location.get("latitude", 0)
                place_lng = location.get("longitude", 0)
                
                distance_miles = _calculate_distance(CAMPGROUND_LAT, CAMPGROUND_LNG, place_lat, place_lng)
                drive_time = _estimate_drive_time(distance_miles)
                
                photo_url = None
                photos = place.get("photos", [])
                if photos:
                    photo_name = photos[0].get("name", "")
                    if photo_name:
                        photo_url = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
                
                place_data = {
                    "name": place.get("displayName", {}).get("text", "Unknown"),
                    "place_id": place_id,
                    "distance_miles": round(distance_miles, 1),
                    "drive_time": drive_time,
                    "rating": place.get("rating"),
                    "review_count": place.get("userRatingCount"),
                    "photo_url": photo_url,
                    "address": place.get("formattedAddress", ""),
                    "types": place.get("types", [])
                }
                results.append(place_data)
                
                if place_id:
                    cache_place(place_id, place_data)
            
            results.sort(key=lambda x: x["distance_miles"])
            
            result_data = {
                "query": query,
                "places_found": len(results),
                "places": results
            }
            cache_place(cache_key, result_data)
            
            return json.dumps(result_data)
            
    except Exception as e:
        return json.dumps({"error": str(e), "places": []})


# =============================================================================
# TRAILS AGENT (Handles ALL trail queries - discovery AND details)
# =============================================================================

trails_agent = Agent(
    name="TrailsAgent",
    description=f"""Use for ALL trail-related queries. Handles both discovery and details.
    
    Examples:
    - "What trails are nearby?" → Discovery mode (compact table)
    - "Where can I hike?" → Discovery mode
    - "Tell me about Wind Wolves Preserve" → Detail mode (rich narrative)
    """,
    model="gemini-2.5-flash",
    instruction=f"""You are the trails expert for {CAMPGROUND_NAME}.

## DISCOVERY: "What trails are nearby?"

Just call discover_all_trails() - it does everything:
- Gets trails from OpenStreetMap (names, surface, coordinates)
- Gets preserves from Google Places (ratings, photos)
- Enriches top OSM trails with Google data using coordinates
- All cached for fast responses

Format as table:
| Trail | Distance | Rating | Surface | Source |
|-------|----------|--------|---------|--------|
| [Tule Elk Trail](maps_url) | 26 mi | ⭐4.8 ℹ️ | unpaved | OSM+Google |
| Wind Wolves Preserve | 23 mi | ⭐4.8 | - | Google |

**Formatting:**
- For OSM trails with maps_url, make the name a markdown link: [Trail Name](maps_url)
- ⭐4.8 ℹ️ = Inferred rating from nearby preserve
- ⭐4.8 = Direct Google rating

## DETAILS: "Tell me about [trail name]"
1. Call get_place_details(name) for photos/rating
2. Call SearchAgent for AllTrails reviews and tips
3. Write a rich narrative (2-3 paragraphs)

## CACHING
All data is cached by region. Repeated queries are instant.
""",
    tools=[
        discover_all_trails,  # Combined OSM + Google with enrichment
        get_place_details,
        AgentTool(agent=search_agent),
    ],
    after_model_callback=after_model_callback,
)


# =============================================================================
# PLACES AGENT (Generic catch-all for restaurants, parks, stores, etc.)
# =============================================================================

places_agent = Agent(
    name="PlacesAgent",
    description=f"""Use for generic place queries (NOT trails). Restaurants, parks, stores, gas stations.
    
    Examples:
    - "What restaurants are nearby?"
    - "Find grocery stores"
    - "Where is the nearest gas station?"
    - "What parks are around here?"
    """,
    model="gemini-2.5-flash",
    instruction=f"""You find places near {CAMPGROUND_NAME}.
Use for restaurants, parks, stores, gas stations - NOT trails.

## YOUR TOOLS

1. **search_nearby_places(query)** - Search for any type of place
   - Pass what user is looking for: "restaurants", "grocery stores", "parks"
   
2. **get_place_details(place_name)** - Get details about a specific place

## OUTPUT FORMAT

| Place | Distance | Rating |
|-------|----------|--------|
| Olive Garden | 5.2 mi | ⭐4.3 (1.2K) |
| Starbucks | 3.1 mi | ⭐4.1 (856) |

Include 1-2 photos of top results.

Keep it simple and quick - just the facts.
""",
    tools=[
        search_nearby_places,
        get_place_details,
    ],
)


# =============================================================================
# ROOT AGENT (Orchestrator)
# =============================================================================

root_agent = Agent(
    name="CampgroundAssistant",
    model="gemini-2.5-flash",
    instruction=f"""You are a helpful assistant for campers staying at {CAMPGROUND_NAME}
in {CAMPGROUND_LOCATION.get('city', 'Bakersfield')}, {CAMPGROUND_LOCATION.get('state', 'CA')}.

## ROUTING

| Query Type | Agent | Examples |
|------------|-------|----------|
| **Trails** | TrailsAgent | "what trails nearby", "where to hike", "tell me about Wind Wolves" |
| **Places** | PlacesAgent | "restaurants nearby", "find grocery stores", "parks around here" |
| **Directions** | render_navigation_card | "how do I get to Starbucks" |
| **Campground** | get_campground_info | "what amenities", "check-out time" |
| **General** | SearchAgent | "weather forecast", "local events" |

## IMPORTANT

- **TrailsAgent** handles ALL trail queries (discovery AND details) - just call it once
- **PlacesAgent** is for generic places (restaurants, stores, parks - NOT trails)
- Pass through agent responses directly - don't modify or summarize them
""",
    tools=[
        AgentTool(agent=trails_agent),   # ALL trail queries
        AgentTool(agent=places_agent),   # Restaurants, stores, parks (not trails)
        render_navigation_card,
        get_campground_info,
        AgentTool(agent=search_agent),
    ],
)


# =============================================================================
# APP SETUP
# =============================================================================

app = App(root_agent=root_agent, name="app")
