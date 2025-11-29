"""
FastAPI server that wraps the ADK agent for the campground frontend.

This server:
1. Serves static HTML files (landing page)
2. Provides /api/campground/{id} endpoint using agent's get_campground_info tool
3. Provides /api/chat endpoint that forwards messages to the ADK agent via SSE
"""

import json
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import the campground data loader from the agent module
from app.agent import CAMPGROUND_DATA, CAMPGROUND_NAME, CAMPGROUND_LAT, CAMPGROUND_LNG

app = FastAPI(title="Campground Agent Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ADK Web Server URL (running on port 8001)
ADK_SERVER_URL = "http://127.0.0.1:5001"

# Google Places API key (for photos proxy)
GOOGLE_PLACES_API_KEY = "AIzaSyB_m4hT4FHoxiZ_JcuHteaVpGbSmwxvmz4"

# Session storage (in-memory for simplicity)
sessions: Dict[str, str] = {}  # user_id -> session_id


async def fetch_place_photos_by_id(place_ids: List[str], client: httpx.AsyncClient) -> Dict[str, str]:
    """
    Fetch photo URLs for a list of place_ids using Google Places API (New).
    Returns a dict mapping place_name -> photo_url
    """
    photos = {}
    
    for place_id in place_ids[:5]:  # Limit to 5 places to avoid too many API calls
        if not place_id:
            continue
            
        # Clean up place_id format (remove 'places/' prefix if present)
        clean_id = place_id.replace('places/', '') if place_id.startswith('places/') else place_id
        
        try:
            # Use Places API (New) to get place details with photos
            url = f"https://places.googleapis.com/v1/places/{clean_id}"
            headers = {
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": "displayName,photos"
            }
            
            response = await client.get(url, headers=headers, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                display_name = data.get("displayName", {}).get("text", "")
                photos_list = data.get("photos", [])
                
                if photos_list and display_name:
                    # Get the first photo's name/reference
                    photo_name = photos_list[0].get("name", "")
                    if photo_name:
                        # Construct photo URL using Places Photo API
                        photo_url = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=800&key={GOOGLE_PLACES_API_KEY}"
                        photos[display_name] = photo_url
                        
        except Exception as e:
            print(f"Error fetching photo for {place_id}: {e}")
            continue
    
    return photos


async def fetch_place_photos_by_name(place_names: List[str], client: httpx.AsyncClient) -> Dict[str, List[str]]:
    """
    Search for places by name and fetch their photos.
    Returns a dict mapping place_name -> [list of photo_urls]
    """
    photos = {}
    
    for place_name in place_names[:3]:  # Limit to 3 to avoid too many API calls
        if not place_name:
            continue
            
        try:
            # Use Places API Text Search to find the place
            url = "https://places.googleapis.com/v1/places:searchText"
            headers = {
                "X-Goog-Api-Key": GOOGLE_PLACES_API_KEY,
                "X-Goog-FieldMask": "places.displayName,places.photos",
                "Content-Type": "application/json"
            }
            body = {
                "textQuery": f"{place_name}, California",
                "maxResultCount": 1
            }
            
            response = await client.post(url, headers=headers, json=body, timeout=5.0)
            
            if response.status_code == 200:
                data = response.json()
                places_list = data.get("places", [])
                
                if places_list:
                    place = places_list[0]
                    display_name = place.get("displayName", {}).get("text", place_name)
                    photos_list = place.get("photos", [])
                    
                    if photos_list:
                        # Get up to 5 photos per place
                        photo_urls = []
                        for photo in photos_list[:5]:
                            photo_name = photo.get("name", "")
                            if photo_name:
                                photo_url = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=1200&key={GOOGLE_PLACES_API_KEY}"
                                photo_urls.append(photo_url)
                        
                        if photo_urls:
                            photos[display_name] = photo_urls
                            # Also add the original search name as a key
                            if display_name.lower() != place_name.lower():
                                photos[place_name] = photo_urls
                        
        except Exception as e:
            print(f"Error searching for place {place_name}: {e}")
            continue
    
    return photos


def extract_place_names_from_response(response_text: str) -> List[str]:
    """
    Extract likely trail/park names from the response text.
    Looks for patterns like "## 1. Place Name" or "**Place Name**"
    """
    place_names = []
    
    # Pattern 1: Markdown headers with numbers like "## 1. Wind Wolves Preserve"
    header_pattern = r'##?\s*\d+\.?\s*([A-Z][^#\n]+?)(?:\n|$)'
    for match in re.finditer(header_pattern, response_text):
        name = match.group(1).strip()
        # Clean up common suffixes
        name = re.sub(r'\s*[-–]\s*.*$', '', name)  # Remove "- description" parts
        if len(name) > 3 and len(name) < 50:
            place_names.append(name)
    
    # Pattern 2: Bold text that looks like place names (capitalized words)
    bold_pattern = r'\*\*([A-Z][A-Za-z\s]+(?:Park|Preserve|Trail|Lake|Canyon|Mountain|Forest|Reserve|Area))\*\*'
    for match in re.finditer(bold_pattern, response_text):
        name = match.group(1).strip()
        if name not in place_names and len(name) > 3:
            place_names.append(name)
    
    return place_names[:5]  # Limit to 5


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    campground_id: str
    message: str
    history: List[ChatMessage] = []
    use_grounding: bool = True


class ChatResponse(BaseModel):
    response: str
    citations: List[Dict] = []
    places: List[Dict] = []
    directions: Optional[Dict] = None
    trails_discovery: Optional[Dict] = None  # Trail discovery with multiple trails
    trail_details: Optional[Dict] = None  # Single trail details
    response_type: str = "text"
    agents_used: List[str] = []  # Track which agents were invoked
    maps_widget_token: Optional[str] = None  # Google Maps contextual widget token
    maps_place_ids: List[str] = []  # Place IDs for fallback rendering
    place_photos: Dict[str, List[str]] = {}  # {place_name: [photo_urls]} from backend


def get_campground_info_direct(category: str) -> dict:
    """Get campground info directly from loaded data."""
    data = CAMPGROUND_DATA

    if category == "basic_info":
        return {
            "name": data.get("name"),
            "facility_id": data.get("facility_id"),
            "location": data.get("location"),
            "contact": data.get("contact"),
            "ratings": data.get("ratings"),
        }
    elif category == "amenities":
        return {
            "amenities": data.get("amenities"),
            "facilities": data.get("descriptions", {}).get("facilities_and_infrastructure"),
        }
    elif category == "site_types":
        descriptions = data.get("descriptions", {})
        return {
            "rv_sites": descriptions.get("facilities_and_infrastructure", {}).get("rv_sites"),
            "tent_sites": descriptions.get("facilities_and_infrastructure", {}).get("tent_sites"),
            "cabins_lodging": descriptions.get("facilities_and_infrastructure", {}).get("cabins_lodging"),
            "campsites": data.get("campsites"),
        }
    elif category == "activities":
        return {
            "on_site": data.get("activities", {}).get("on_site"),
            "nearby": data.get("activities", {}).get("nearby"),
            "recreation": data.get("descriptions", {}).get("recreation_opportunities"),
        }
    elif category == "all":
        return data
    else:
        return {"error": f"Unknown category: {category}"}


def flatten_amenities(amenities_data) -> List[str]:
    """Flatten amenities dict to a list of strings."""
    amenities_list = []
    if isinstance(amenities_data, dict):
        for category, items in amenities_data.items():
            if isinstance(items, dict):
                amenities_list.extend(items.keys())
            elif isinstance(items, list):
                amenities_list.extend(items)
    elif isinstance(amenities_data, list):
        amenities_list = amenities_data
    return [a.replace("_", " ").title() for a in amenities_list[:15]]


def flatten_activities(activities_data) -> List[str]:
    """Flatten activities dict to a list of strings."""
    activities_list = []
    if isinstance(activities_data, dict):
        on_site = activities_data.get("on_site", [])
        if isinstance(on_site, list):
            for a in on_site:
                if isinstance(a, dict) and a.get("name"):
                    activities_list.append(a.get("name"))
                elif isinstance(a, str):
                    activities_list.append(a)
    elif isinstance(activities_data, list):
        activities_list = activities_data
    return activities_list[:10]


def get_site_types(campsites_data) -> List[str]:
    """Extract site type names from campsites data."""
    if not campsites_data:
        return []
    types = campsites_data.get("types", [])
    return [s.get("name") for s in types if s.get("name")]


async def get_or_create_session(user_id: str) -> str:
    """Get existing session or create new one with ADK server."""
    if user_id in sessions:
        return sessions[user_id]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ADK_SERVER_URL}/apps/app/users/{user_id}/sessions",
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            session_data = response.json()
            session_id = session_data.get("id")
            sessions[user_id] = session_id
            return session_id
        else:
            raise HTTPException(status_code=500, detail="Failed to create ADK session")


async def send_to_agent(user_id: str, message: str) -> dict:
    """Send a message to the ADK agent and get the response."""
    session_id = await get_or_create_session(user_id)

    request_body = {
        "app_name": "app",
        "user_id": user_id,
        "session_id": session_id,
        "new_message": {
            "role": "user",
            "parts": [{"text": message}]
        }
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{ADK_SERVER_URL}/run_sse",
            json=request_body,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Agent request failed")

        # Parse SSE response
        full_response = ""
        agents_used = set()
        tools_used = set()
        
        # Maps grounding data from customMetadata (clean approach!)
        maps_widget_token = None
        maps_places = []
        maps_place_ids = []
        
        # Tool-based structured responses
        navigation_directions = None

        for line in response.text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])

                    # Track the author (agent name)
                    author = data.get("author")
                    if author:
                        agents_used.add(author)

                    # Extract maps grounding data from stateDelta (flows through SSE from sub-agents)
                    actions = data.get("actions", {})
                    state_delta = actions.get("stateDelta", {})
                    if state_delta.get("maps_grounding"):
                        grounding_data = state_delta["maps_grounding"]
                        if grounding_data.get("widget_token"):
                            maps_widget_token = grounding_data["widget_token"]
                        if grounding_data.get("places"):
                            maps_places = grounding_data["places"]
                        if grounding_data.get("place_ids"):
                            maps_place_ids = grounding_data["place_ids"]

                    # Check for function calls and responses
                    content = data.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        # Track function calls
                        if "functionCall" in part:
                            func_name = part["functionCall"].get("name", "")
                            if func_name:
                                tools_used.add(func_name)

                        # Extract data from functionResponse (for custom tools)
                        if "functionResponse" in part:
                            func_resp = part["functionResponse"]
                            func_name = func_resp.get("name", "")
                            resp_data = func_resp.get("response", {})
                            result = resp_data.get("result", "")
                            
                            # Handle render_navigation_card tool
                            if func_name == "render_navigation_card" and result:
                                try:
                                    nav_data = json.loads(result)
                                    if nav_data.get("type") == "navigation":
                                        navigation_directions = {
                                            "origin": nav_data.get("origin", "user_current_location"),
                                            "destination": nav_data.get("destination", ""),
                                            "travel_mode": nav_data.get("travel_mode", "DRIVING"),
                                            "campground_location": nav_data.get("campground_location", {})
                                        }
                                except json.JSONDecodeError:
                                    pass
                            
                            # Note: render_trails_discovery and render_trail_details removed
                            # Discovery now uses PlacesAgent + SearchAgent directly

                        # Get text from response
                        if "text" in part:
                            full_response = part["text"]

                except json.JSONDecodeError:
                    continue

        # Combine agents and tools for display
        all_sources = list(agents_used | tools_used)

        # Clean up response text - remove [RESPONSE_TYPE: ...] tags
        clean_response = re.sub(r'\s*\[RESPONSE_TYPE:\s*\w+\]\s*', '', full_response).strip()
        
        # Determine response type
        response_type = "text"
        if navigation_directions:
            response_type = "navigation"
        elif maps_widget_token or maps_places:
            response_type = "places"

        # Fetch photos for places (backend approach)
        place_photos = {}
        
        # Method 1: Use place_ids if available (from Maps grounding)
        if maps_place_ids:
            place_photos = await fetch_place_photos_by_id(maps_place_ids, client)
        
        # Method 2: Extract place names from response and search (for search grounding)
        if not place_photos and clean_response:
            place_names = extract_place_names_from_response(clean_response)
            print(f"[Photo Fetch] Extracted place names: {place_names}", flush=True)
            if place_names:
                try:
                    place_photos = await fetch_place_photos_by_name(place_names, client)
                    print(f"[Photo Fetch] Got photos for: {list(place_photos.keys())}", flush=True)
                except Exception as e:
                    print(f"[Photo Fetch] Error: {e}", flush=True)
                    place_photos = {}

        # Ensure place_photos is always a dict
        if place_photos is None:
            place_photos = {}
        
        print(f"[Photo Fetch] FINAL place_photos before return: {list(place_photos.keys())}", flush=True)
            
        return {
            "response": clean_response,
            "citations": [],
            "places": maps_places,
            "directions": navigation_directions,
            "trails_discovery": None,  # Deprecated - discovery uses markdown now
            "trail_details": None,     # Deprecated - details use markdown now
            "response_type": response_type,
            "agents_used": all_sources,
            "maps_widget_token": maps_widget_token,
            "maps_place_ids": maps_place_ids,
            "place_photos": place_photos  # {place_name: photo_url}
        }


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/api/campground/{campground_id}")
async def get_campground(campground_id: str):
    """Get campground data for the frontend."""
    data = CAMPGROUND_DATA
    location = data.get("location", {})

    # Build response matching the lmtestbed format
    return {
        "id": data.get("facility_id"),
        "name": data.get("name"),
        "location": location,
        "images": [],  # Will be populated from Google Places or fallback
        "amenities": flatten_amenities(data.get("amenities")),
        "activities": flatten_activities(data.get("activities")),
        "site_types": get_site_types(data.get("campsites")),
        "editorial": {
            "quick_summary": {
                "Style": "RV Resort",
                "Best For": "Road Trippers & Families",
                "Sites": "Full Hookup Pull-Throughs",
                "Access": "Easy Freeway Access",
                "Shade": "Limited",
                "Pets": "Dog Friendly",
                "Noise": "Light Urban Background",
                "Vibe": "Polished & Convenient"
            },
            "atmosphere": data.get("descriptions", {}).get("overview", ""),
            "who_loves": {
                "pros": ["Big-rig travelers", "Families needing a pool", "Road trippers"],
                "cons": ["Long scenic stays", "Those seeking wilderness solitude"]
            }
        }
    }


@app.get("/api/campground/{campground_id}/markdown")
async def get_campground_markdown(campground_id: str):
    """Get campground data with markdown editorial content."""
    data = CAMPGROUND_DATA
    location = data.get("location", {})

    # Generate a simple markdown overview
    overview = data.get("descriptions", {}).get("overview", "")

    return {
        "id": data.get("facility_id"),
        "name": data.get("name"),
        "location": location,
        "images": [],
        "markdown": f"# Welcome to {data.get('name')}\n\n{overview}"
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint that forwards messages to the ADK agent."""
    user_id = "frontend_user"

    try:
        result = await send_to_agent(user_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint - returns SSE with progressive updates."""
    from fastapi.responses import StreamingResponse
    
    user_id = "frontend_user"
    
    async def generate_stream():
        session_id = await get_or_create_session(user_id)
        
        request_body = {
            "app_name": "app",
            "user_id": user_id,
            "session_id": session_id,
            "new_message": {
                "role": "user",
                "parts": [{"text": request.message}]
            }
        }
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{ADK_SERVER_URL}/run_sse",
                json=request_body,
                headers={"Accept": "text/event-stream"}
            ) as response:
                
                current_agent = None
                current_status = None
                accumulated_text = ""
                place_ids = []
                
                # Agent display names
                AGENT_DISPLAYS = {
                    "TrailsAgent": "🥾 Finding trails...",
                    "PlacesAgent": "📍 Finding places...",
                    "SearchAgent": "🔍 Searching the web...",
                    "CampgroundAssistant": "💭 Thinking...",
                }
                
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    
                    try:
                        data = json.loads(line[6:])
                        
                        # Detect agent being called via functionCall
                        if "content" in data and data["content"]:
                            parts = data["content"].get("parts", [])
                            for part in parts:
                                # Detect agent routing (root calling sub-agent)
                                if "functionCall" in part:
                                    agent_name = part["functionCall"].get("name", "")
                                    if agent_name in AGENT_DISPLAYS and agent_name != current_agent:
                                        current_agent = agent_name
                                        status = AGENT_DISPLAYS[agent_name]
                                        if status != current_status:
                                            current_status = status
                                            yield f"data: {json.dumps({'type': 'agent', 'agent': status})}\n\n"
                        
                        # Detect author changes (which agent is responding)
                        if "author" in data:
                            author = data["author"]
                            if author and author != "CampgroundAssistant":
                                # Sub-agent is responding
                                if author in AGENT_DISPLAYS and author != current_agent:
                                    current_agent = author
                                    # Show contextual status based on agent
                                    if author == "TrailsAgent":
                                        status = "🥾 Finding trails..."
                                    elif author == "PlacesAgent":
                                        status = "📍 Finding places..."
                                    else:
                                        status = AGENT_DISPLAYS.get(author, f"Using {author}...")
                                    
                                    if status != current_status:
                                        current_status = status
                                        yield f"data: {json.dumps({'type': 'agent', 'agent': status})}\n\n"
                        
                        # Detect google_search tool being used (grounding)
                        if "groundingMetadata" in str(data) or "searchEntryPoint" in str(data):
                            if current_status != "🔍 Searching Google...":
                                current_status = "🔍 Searching Google..."
                                yield f"data: {json.dumps({'type': 'status', 'status': current_status})}\n\n"
                        
                        # Extract grounding metadata for place_ids
                        if "actions" in data and "stateDelta" in data["actions"]:
                            state_delta = data["actions"]["stateDelta"]
                            if "maps_grounding" in state_delta:
                                grounding = state_delta["maps_grounding"]
                                place_ids = grounding.get("place_ids", [])
                        
                        # Stream text content
                        if "content" in data and data["content"]:
                            parts = data["content"].get("parts", [])
                            for part in parts:
                                if "text" in part:
                                    text_chunk = part["text"]
                                    accumulated_text += text_chunk
                                    yield f"data: {json.dumps({'type': 'text', 'text': text_chunk})}\n\n"
                    
                    except json.JSONDecodeError:
                        continue
                
                # After streaming complete, fetch photos and send final metadata
                clean_response = re.sub(r'\s*\[RESPONSE_TYPE:\s*\w+\]\s*', '', accumulated_text).strip()
                
                # Extract place names and fetch photos
                place_photos = {}
                if clean_response:
                    place_names = extract_place_names_from_response(clean_response)
                    if place_names:
                        async with httpx.AsyncClient(timeout=30.0) as photo_client:
                            place_photos = await fetch_place_photos_by_name(place_names, photo_client)
                
                # Send final metadata
                yield f"data: {json.dumps({'type': 'done', 'place_photos': place_photos})}\n\n"
        
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/place-photo/{photo_reference}")
async def proxy_place_photo(photo_reference: str, maxwidth: int = 400):
    """Proxy endpoint for Google Places photos."""
    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={maxwidth}&photo_reference={photo_reference}&key={GOOGLE_PLACES_API_KEY}"

    async with httpx.AsyncClient(follow_redirects=True) as client:
        response = await client.get(photo_url)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail="Failed to fetch photo")

        from fastapi.responses import Response
        return Response(
            content=response.content,
            media_type=response.headers.get("content-type", "image/jpeg"),
            headers={"Cache-Control": "public, max-age=86400"}
        )


# =============================================================================
# Static Files and HTML
# =============================================================================

@app.get("/")
async def root():
    """Serve the landing page."""
    landing_page = Path(__file__).parent / "static" / "index.html"
    if landing_page.exists():
        return FileResponse(landing_page)
    else:
        return {"message": "Landing page not found. Create static/index.html"}


# Mount static files
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)
