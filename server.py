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
ADK_SERVER_URL = "http://127.0.0.1:8001"

# Google Places API key (for photos proxy)
GOOGLE_PLACES_API_KEY = "AIzaSyB_m4hT4FHoxiZ_JcuHteaVpGbSmwxvmz4"

# Session storage (in-memory for simplicity)
sessions: Dict[str, str] = {}  # user_id -> session_id


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
    response_type: str = "text"
    agents_used: List[str] = []  # Track which agents were invoked
    maps_widget_token: Optional[str] = None  # Google Maps contextual widget token


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

        # Parse SSE response - collect all text from model responses and track agents
        full_response = ""
        agents_used = set()  # Track which agents were invoked
        tools_used = set()  # Track which tools were called
        maps_widget_token = None  # Extract from functionResponse

        for line in response.text.split("\n"):
            if line.startswith("data: "):
                try:
                    data = json.loads(line[6:])

                    # Track the author (agent name)
                    author = data.get("author")
                    if author:
                        agents_used.add(author)

                    # Check for function calls (sub-agent invocations)
                    content = data.get("content", {})
                    parts = content.get("parts", [])
                    for part in parts:
                        # Track function calls to sub-agents
                        if "functionCall" in part:
                            func_name = part["functionCall"].get("name", "")
                            if func_name:
                                tools_used.add(func_name)

                        # Extract widget token from functionResponse (MapsAgent output)
                        # This is where the callback-injected token lives
                        if "functionResponse" in part:
                            func_resp = part["functionResponse"]
                            resp_data = func_resp.get("response", {})
                            result = resp_data.get("result", "")
                            if "[MAPS_WIDGET_DATA:" in result and not maps_widget_token:
                                # Extract the widget token directly from functionResponse
                                widget_match = re.search(r'\[MAPS_WIDGET_DATA:(\{.*?"grounding_chunks":\s*\[\]\})', result, re.DOTALL)
                                if widget_match:
                                    try:
                                        widget_json_str = widget_match.group(1)
                                        widget_json_str = widget_json_str.replace('\\"', '"').replace('\\n', '\n')
                                        widget_data = json.loads(widget_json_str)
                                        maps_widget_token = widget_data.get("widget_token")
                                        print(f"DEBUG: Extracted widget token from functionResponse: {maps_widget_token[:50]}...")
                                    except json.JSONDecodeError as e:
                                        print(f"DEBUG: Failed to parse widget JSON from functionResponse: {e}")

                        # Get text from final response
                        if "text" in part:
                            full_response = part["text"]  # Take the last text response

                except json.JSONDecodeError:
                    continue

        # Combine agents and tools for display
        all_sources = list(agents_used | tools_used)

        # Try to parse the response as JSON (from MapsAgent)
        places = []
        directions = None
        response_type = "text"  # Default fallback
        # maps_widget_token already extracted from functionResponse above

        # First, check if response is an ADK AgentTool wrapper like {"MapsAgent_response": {"result": "..."}}
        # Use regex to extract the result field to avoid JSON escape issues
        wrapper_match = re.search(r'"(\w+Agent)_response":\s*\{\s*"result":\s*"(.*)"', full_response, re.DOTALL)
        if wrapper_match:
            agent_name = wrapper_match.group(1)
            result_str = wrapper_match.group(2)
            # Unescape the result string
            result_str = result_str.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            full_response = result_str
            # Clean up any "json\n" prefix from markdown formatting
            if full_response.startswith('json\n'):
                full_response = full_response[5:]
            # Remove widget marker from result
            if "[MAPS_WIDGET_DATA:" in full_response:
                full_response = re.sub(r'\s*\[MAPS_WIDGET_DATA:\{.*?"grounding_chunks":\s*\[\]\}\s*', '', full_response, flags=re.DOTALL).strip()
            print(f"DEBUG: Unwrapped {agent_name} response (first 200): {full_response[:200]}")

        # Widget token already extracted from functionResponse in SSE parsing above
        # Just clean up any marker that might be in the response text
        if "[MAPS_WIDGET_DATA:" in full_response:
            full_response = re.sub(r'\s*\[MAPS_WIDGET_DATA:\{.*?\}\]\s*', '', full_response, flags=re.DOTALL).strip()

        # Extract JSON from markdown code block if present (handles nested code blocks too)
        json_to_parse = full_response

        # Try to find the innermost JSON with "type" field
        # First look for ```json blocks
        json_matches = re.findall(r'```json\s*([\s\S]*?)\s*```', full_response)
        for match in json_matches:
            # Try to find a valid places/directions JSON
            try:
                test_json = json.loads(match.strip())
                if isinstance(test_json, dict) and "type" in test_json:
                    json_to_parse = match.strip()
                    break
            except:
                continue

        # If no markdown block, check for raw JSON with type field
        if json_to_parse == full_response:
            # Check if the response starts with { and contains "type"
            if full_response.strip().startswith('{') and '"type"' in full_response:
                json_to_parse = full_response.strip()

        # Try to parse as JSON
        try:
            parsed_json = json.loads(json_to_parse)
            if isinstance(parsed_json, dict) and "type" in parsed_json:
                response_type = parsed_json["type"]
                if response_type == "places":
                    places = parsed_json.get("places", [])
                    # Keep summary as response text
                    full_response = parsed_json.get("summary", "")
                elif response_type == "directions":
                    directions = {
                        "summary": parsed_json.get("summary", ""),
                        "origin": parsed_json.get("origin", ""),
                        "destination": parsed_json.get("destination", ""),
                        "distance": parsed_json.get("distance", ""),
                        "duration": parsed_json.get("duration", ""),
                        "steps": parsed_json.get("steps", [])
                    }
                    full_response = parsed_json.get("summary", "")
        except (json.JSONDecodeError, TypeError):
            # Not JSON, check for [RESPONSE_TYPE: ...] tag
            response_type_match = re.search(r'\[RESPONSE_TYPE:\s*(\w+)\]', full_response)
            if response_type_match:
                response_type = response_type_match.group(1).lower()
                # Remove the response type tag from the displayed response
                full_response = re.sub(r'\s*\[RESPONSE_TYPE:\s*\w+\]\s*', '', full_response).strip()

        return {
            "response": full_response,
            "citations": [],
            "places": places,
            "directions": directions,
            "response_type": response_type,
            "agents_used": all_sources,
            "maps_widget_token": maps_widget_token
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
    # Use a simple user_id for now
    user_id = "frontend_user"

    try:
        result = await send_to_agent(user_id, request.message)
        return ChatResponse(**result)
    except Exception as e:
        print(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
