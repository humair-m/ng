import os
import time
import logging
import subprocess
import json
import uuid
import base64
import requests
import concurrent.futures
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from pathlib import Path
import hashlib
import mimetypes
from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Query, Path as FastAPIPath, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pyngrok import ngrok
import uvicorn
import threading
import psutil

# -------------------------
# 🔧 Configuration
# -------------------------
PORT = 5000
UPLOAD_FOLDER = "/home/humair/_ai_/received_images"
ANALYSIS_FOLDER = "/home/humair/_ai_/analysis_results"
API_KEY = "mysecurekey123"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
RATE_LIMIT = 60  # requests per minute

# AI Analysis API Configuration
AI_API_URL = "https://text.pollinations.ai/openai"
AI_MODEL = "o4-mini"
AI_MAX_TOKENS = 5500
AI_RETRIES = 5  # Max retries for AI API calls
AI_RETRY_DELAY_BASE = 2  # Base delay in seconds for exponential backoff

# -------------------------
# 📊 Models
# -------------------------
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: str
    version: str
    disk_usage: Dict[str, Any]
    memory_usage: Dict[str, Any]

class FileInfo(BaseModel):
    filename: str
    original_name: str
    size: int
    upload_time: str
    file_hash: str
    mime_type: str
    analysis_status: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    status: str
    filename: str
    file_id: str
    size: int
    upload_time: str
    analysis_status: str
    analysis_result: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    status: str
    filename: str
    analysis: Dict[str, Any]
    analysis_time: str

class StatsResponse(BaseModel):
    total_files: int
    total_size: int
    upload_folder_size: int
    analyzed_files: int
    pending_analysis: int
    oldest_file: Optional[str]
    newest_file: Optional[str]
    file_types: Dict[str, int]

# -------------------------
# 🧾 Logging Setup
# -------------------------
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANALYSIS_FOLDER, exist_ok=True)

log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger("ImageReceiver")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("image_receiver.log")
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

# -------------------------
# 🚀 FastAPI App
# -------------------------
app = FastAPI(
    title="Enhanced Image Server with AI Analysis",
    description="A comprehensive image upload, analysis, and management server",
    version="3.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for tracking
start_time = time.time()
upload_stats = {
    "total_uploads": 0,
    "total_size": 0,
    "analyzed_files": 0,
    "pending_analysis": 0,
    "last_upload": None
}

# In-memory storage for analysis results
analysis_results = {}

# -------------------------
# 🔐 Authentication Helper
# -------------------------
def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        logger.warning(f"🚫 Invalid API key attempt: {x_api_key}")
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

# -------------------------
# 🛠️ Utility Functions
# -------------------------
def get_file_hash(filepath: str) -> str:
    """Generate MD5 hash of file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def get_folder_size(folder_path: str) -> int:
    """Calculate total size of folder"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            total_size += os.path.getsize(filepath)
    return total_size

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image_with_ai(image_path: str, filename: str) -> Dict[str, Any]:
    """Send image to AI API for analysis with retries"""
    encoded_image = encode_image_to_base64(image_path)

    # Prepare the request payload
    payload = {
        "model": AI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Conduct an exhaustive visual analysis of the image in extreme detail, with a word count strictly between "
                            "**2500 to 3800 words**. Maintain an academic and formal tone throughout. "
                            "the **completeness, accuracy, and specificity are critical**.\n\n"

                            "**KEY DIRECTIVES:**\n"
                            "- Focus solely on what **is visible** in the image — do **NOT** speculate or comment on what is absent.\n"
                            "- If applicable, provide relevant **scientific explanations** (biological, chemical, physical, mathematical, "
                            "psychological, social, etc.) for the phenomena, events, or configurations observed in the image.Show if related to romance , hate , ...else \n"
                            "- Identify and articulate the image's **theme, artistic intention, narrative purpose, or communicative objective**, "
                            "based on visual evidence.\n\n"

                            "**METICULOUSLY DESCRIBE THE FOLLOWING CATEGORIES IN PARAGRAPHS:**\n\n"

                            "1) **All Visible Objects and Spatial Relationships**:IN DETAIL\n"
                            "   - List and count all distinguishable objects (**always provide EXACT number or best estimate**).\n"
                            "   - Describe how they are positioned in relation to each other (foreground, background, left/right, overlapping, etc.).\n\n"

                            "2) **People** (if present):\n"
                            "   - Detail their postures, clothing, hairstyle, visible accessories, apparent ethnicity (if visually discernible), "
                            "age group (child, adult, elderly), and any visible deformities or health conditions.\n\n"

                            "3) **Actions, Gestures, and Interactions**:\n"
                            "   - Explain what each person or object is doing or interacting with, and describe dynamic movements or expressions.\n\n"

                            "4) **Background Elements**:\n"
                            "   - Include any visible natural or artificial background features (e.g., buildings, trees, decor, sky, textural backdrops).\n\n"

                            "5) **Lighting, Shadows, and Reflections**IN DEPTH:\n"
                            "   - Describe the type (natural vs artificial), source direction, softness or harshness, and its effects on shadows and reflections. "
                            "**(Skip if image is a presentation or notebook page with only text).**\n\n"

                            "6) **Colors, Contrast, and Palette Distribution**:\n"
                            "   - Analyze dominant and secondary color schemes, saturation, tonal balance, and their visual/emotional impact.\n\n"

                            "7) **Textures and Surface Quality**:\n"
                            "   - Describe the visual textures of surfaces, materials, fabrics, skin, and objects. "
                            "**(Skip if image is a presentation or notebook page with only text).**\n\n"

                            "8) **Camera/Viewer Perspective**:\n"
                            "   - Discuss the camera angle (low, high, eye-level), framing (portrait, landscape, close-up, wide shot), "
                            "focus depth (shallow or deep), and overall composition.\n\n"

                            "9) **Artistic Style and Symbolism** IN DEPTH :\n"
                            "   - Identify the visual art style (realism, surrealism, abstract, etc.), any symbolic elements, visual metaphors, or allegories.\n\n"

                            "10) **Visible Text (if any)**:\n"
                            "   - Transcribe ALL visible text **verbatim**, regardless of length. Provide **translations** if the text is not in English. "
                            "Do not summarize — **return the entire content as-is**.\n\n"

                            "11) **Cultural or Historical Indicators**:\n"
                            "   - If relevant, infer cultural/historical context from visible architecture, costumes, flora, artifacts, or visual motifs. "
                            "**(Skip if image is a presentation or notebook page with only text).**\n\n"

                            "12) **Mood or Emotional Tone**:\n"
                            "   - Assess the emotional atmosphere (e.g., joy, sadness, neutrality, tension, sarcasm, affection, romance). "
                            "Justify your interpretation with visual evidence.\n\n"

                            "13) **What Is Happening Right Now** IN DEPTH :\n"
                            "   - Offer a detailed narrative of the ongoing event, action, or moment captured in the image.\n\n"

                            "14) **Scientific Topic or Concept Diagram**:\n"
                            "   - If the image includes a visual representation of a scientific concept or process (e.g., physics diagram, biochemical pathway), "
                            "describe and explain it thoroughly. **If this is not present, omit this point entirely.**\n\n"

                            "15) **Special Handling for Presentation or Notebook Text Pages**:\n"
                            "   - If the image contains only text (e.g., a handwritten or slide-based page, book page), SKIP points 5, 7, and 11 entirely.\n\n"

                            "16) **Humor, Meme, or Joke Recognition**:\n"
                            "   - If the image is a meme or contains humor, jokes, or satire, identify it clearly. Explain:\n"
                            "     a) Why it is humorous or meme-like,\n"
                            "     b) Cultural or social context needed to understand it,\n"
                            "     c) Structure or formula of the humor.\n"
                            "   **If humor is not present, omit this point entirely.**\n\n"

                            "17) **Statistical Data, Graphs, or Charts**:\n"
                            "   - If the image contains any charts, graphs, or statistical visuals, interpret all data accurately,"
                                "explaining axes, trends, numerical values, and implications.\n "
                            "18) if animal/plant/species  then tell their scientific name ,where they found , there chaactersticks , appeareance"

                            "#Warning: Don't skip any section unless the specified skip conditions are fully met."
                            "#Detaied , by keeping all things in context"
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": AI_MAX_TOKENS
    }




    for attempt in range(AI_RETRIES):
        try:
            logger.info(f"🔍 Starting AI analysis for: {filename} (Attempt {attempt + 1}/{AI_RETRIES})")
            
            response = requests.post(
                AI_API_URL, 
                headers={"Content-Type": "application/json"}, 
                data=json.dumps(payload),
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ AI analysis completed for: {filename} on attempt {attempt + 1}")
                return {
                    "status": "success",
                    "analysis": result,
                    "analysis_time": datetime.now().isoformat(),
                    "model_used": AI_MODEL
                }
            elif response.status_code in [429, 500, 502, 503, 504]:  # Retryable errors (Too Many Requests, Server Errors)
                if attempt < AI_RETRIES - 1:
                    wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"⚠️ AI API error for {filename}: {response.status_code} - {response.text}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ AI API error for {filename} after {AI_RETRIES} attempts: {response.status_code} - {response.text}")
                    return {
                        "status": "error",
                        "error": f"API Error {response.status_code}: {response.text}",
                        "analysis_time": datetime.now().isoformat()
                    }
            else:
                logger.error(f"❌ Non-retryable AI API error for {filename}: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "error": f"API Error {response.status_code}: {response.text}",
                    "analysis_time": datetime.now().isoformat()
                }
                
        except requests.exceptions.Timeout:
            if attempt < AI_RETRIES - 1:
                wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(f"⏰ AI API request timed out for {filename}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ AI API request timed out for {filename} after {AI_RETRIES} attempts.")
                return {
                    "status": "error",
                    "error": "AI API request timed out",
                    "analysis_time": datetime.now().isoformat()
                }
        except requests.exceptions.RequestException as e:
            if attempt < AI_RETRIES - 1:
                wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(f"🌐 Network error during AI analysis for {filename}: {str(e)}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"❌ Network error during AI analysis for {filename} after {AI_RETRIES} attempts: {str(e)}")
                return {
                    "status": "error",
                    "error": f"Network error: {str(e)}",
                    "analysis_time": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"❌ Unexpected error during analysis for {filename}: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_time": datetime.now().isoformat()
            }
    
    # This part should ideally not be reached if all attempts fail, but as a fallback:
    return {
        "status": "error",
        "error": f"Failed to get AI analysis after {AI_RETRIES} attempts.",
        "analysis_time": datetime.now().isoformat()
    }

def save_analysis_to_json(filename: str, analysis_result: Dict[str, Any], file_info: Dict[str, Any]):
    """Save analysis result and file info to JSON"""
    try:
        analysis_data = {
            "filename": filename,
            "file_info": file_info,
            "analysis_result": analysis_result,
            "saved_time": datetime.now().isoformat()
        }
        
        # Save individual analysis file
        analysis_filename = f"{Path(filename).stem}_analysis.json"
        analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
        
        with open(analysis_filepath, "w") as f:
            json.dump(analysis_data, f, indent=2)
        
        # Update master analysis file
        master_file = os.path.join(ANALYSIS_FOLDER, "all_analyses.json")
        if os.path.exists(master_file):
            with open(master_file, "r") as f:
                all_analyses = json.load(f)
        else:
            all_analyses = []
        
        all_analyses.append(analysis_data)
        
        with open(master_file, "w") as f:
            json.dump(all_analyses, f, indent=2)
        
        logger.info(f"💾 Analysis saved to JSON for: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error saving analysis to JSON: {str(e)}")

def process_image_analysis(filepath: str, filename: str, file_info: Dict[str, Any]):
    """Background task to analyze image"""
    try:
        upload_stats["pending_analysis"] += 1
        
        # Perform AI analysis
        analysis_result = analyze_image_with_ai(filepath, filename)
        
        # Store in memory
        analysis_results[filename] = analysis_result
        
        # Save to JSON
        save_analysis_to_json(filename, analysis_result, file_info)
        
        # Update stats
        upload_stats["pending_analysis"] = max(0, upload_stats["pending_analysis"] - 1)
        if analysis_result["status"] == "success":
            upload_stats["analyzed_files"] += 1
        
        logger.info(f"🎯 Analysis pipeline completed for: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error in analysis pipeline for {filename}: {str(e)}")
        upload_stats["pending_analysis"] = max(0, upload_stats["pending_analysis"] - 1)

# -------------------------
# 📡 API Endpoints
# -------------------------

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Enhanced Image Server with AI Analysis",
        "version": "3.0.0",
        "features": ["Upload", "AI Analysis", "File Management", "Statistics"],
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    uptime = time.time() - start_time
    uptime_str = str(timedelta(seconds=int(uptime)))
    
    # Disk usage
    disk_usage = psutil.disk_usage(os.getcwd())
    disk_info = {
        "total": disk_usage.total,
        "used": disk_usage.used,
        "free": disk_usage.free,
        "percent": (disk_usage.used / disk_usage.total) * 100
    }
    
    # Memory usage
    memory = psutil.virtual_memory()
    memory_info = {
        "total": memory.total,
        "used": memory.used,
        "available": memory.available,
        "percent": memory.percent
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        uptime=uptime_str,
        version="3.0.0",
        disk_usage=disk_info,
        memory_usage=memory_info
    )

@app.post("/upload", response_model=UploadResponse)
async def upload_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    x_api_key: str = Header(None),
    analyze: bool = Query(True, description="Whether to analyze the image with AI")
):
    """Upload an image file with optional AI analysis"""
    verify_api_key(x_api_key)
    
    logger.info(f"📥 Incoming file: {file.filename}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not is_allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File type not allowed")
    
    # Read file content
    content = await file.read()
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    extension = Path(file.filename).suffix
    filename = f"{timestamp}_{file_id}{extension}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    try:
        with open(filepath, "wb") as f:
            f.write(content)
        
        # Update stats
        upload_stats["total_uploads"] += 1
        upload_stats["total_size"] += len(content)
        upload_stats["last_upload"] = datetime.now().isoformat()
        
        # File info for analysis
        file_info = {
            "filename": filename,
            "original_name": file.filename,
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
            "file_hash": get_file_hash(filepath),
            "mime_type": mimetypes.guess_type(filepath)[0] or "application/octet-stream"
        }
        
        logger.info(f"✅ Saved to: {filepath}")
        
        # Prepare response
        response_data = {
            "status": "success",
            "filename": filename,
            "file_id": file_id,
            "size": len(content),
            "upload_time": datetime.now().isoformat(),
            "analysis_status": "not_requested"
        }
        
        # Start AI analysis if requested
        if analyze:
            response_data["analysis_status"] = "processing"
            background_tasks.add_task(process_image_analysis, filepath, filename, file_info)
        
        return UploadResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/analyze/{filename}", response_model=AnalysisResponse)
async def get_analysis(
    filename: str = FastAPIPath(...),
    x_api_key: str = Header(None)
):
    """Get analysis result for a specific file"""
    verify_api_key(x_api_key)
    
    # Check if analysis exists in memory
    if filename in analysis_results:
        return AnalysisResponse(
            status="success",
            filename=filename,
            analysis=analysis_results[filename],
            analysis_time=analysis_results[filename].get("analysis_time", "")
        )
    
    # Check if analysis exists in JSON files
    analysis_filename = f"{Path(filename).stem}_analysis.json"
    analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
    
    if os.path.exists(analysis_filepath):
        try:
            with open(analysis_filepath, "r") as f:
                analysis_data = json.load(f)
            
            return AnalysisResponse(
                status="success",
                filename=filename,
                analysis=analysis_data["analysis_result"],
                analysis_time=analysis_data["analysis_result"].get("analysis_time", "")
            )
        except Exception as e:
            logger.error(f"❌ Error reading analysis file: {e}")
    
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.post("/analyze/{filename}")
async def analyze_existing_file(
    background_tasks: BackgroundTasks,
    filename: str = FastAPIPath(...),
    x_api_key: str = Header(None)
):
    """Analyze an existing uploaded file"""
    verify_api_key(x_api_key)
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Prepare file info
    stat = os.stat(filepath)
    file_info = {
        "filename": filename,
        "original_name": filename.split('_', 2)[-1] if '_' in filename else filename,
        "size": stat.st_size,
        "upload_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "file_hash": get_file_hash(filepath),
        "mime_type": mimetypes.guess_type(filepath)[0] or "application/octet-stream"
    }
    
    # Start analysis
    background_tasks.add_task(process_image_analysis, filepath, filename, file_info)
    
    return {
        "status": "success",
        "message": f"Analysis started for {filename}",
        "filename": filename
    }

@app.get("/files", response_model=List[FileInfo])
async def list_files(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    include_analysis: bool = Query(True, description="Include analysis status"),
    x_api_key: str = Header(None)
):
    """List uploaded files with pagination and analysis status"""
    verify_api_key(x_api_key)
    
    files = []
    for filename in sorted(os.listdir(UPLOAD_FOLDER))[offset:offset + limit]:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            file_info = FileInfo(
                filename=filename,
                original_name=filename.split('_', 2)[-1] if '_' in filename else filename,
                size=stat.st_size,
                upload_time=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                file_hash=get_file_hash(filepath),
                mime_type=mimetypes.guess_type(filepath)[0] or "application/octet-stream"
            )
            
            # Add analysis status if requested
            if include_analysis:
                if filename in analysis_results:
                    file_info.analysis_status = analysis_results[filename]["status"]
                    file_info.analysis_result = analysis_results[filename]
                else:
                    # Check JSON file
                    analysis_filename = f"{Path(filename).stem}_analysis.json"
                    analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
                    if os.path.exists(analysis_filepath):
                        file_info.analysis_status = "completed"
                    else:
                        file_info.analysis_status = "not_analyzed"
            
            files.append(file_info)
    
    return files

@app.get("/stats", response_model=StatsResponse)
async def get_stats(x_api_key: str = Header(None)):
    """Get comprehensive server statistics"""
    verify_api_key(x_api_key)
    
    files = os.listdir(UPLOAD_FOLDER)
    total_files = len(files)
    
    if total_files == 0:
        return StatsResponse(
            total_files=0,
            total_size=0,
            upload_folder_size=0,
            analyzed_files=0,
            pending_analysis=0,
            oldest_file=None,
            newest_file=None,
            file_types={}
        )
    
    # Calculate stats
    file_stats = []
    file_types = {}
    analyzed_count = 0
    
    for filename in files:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(filepath):
            stat = os.stat(filepath)
            file_stats.append((filename, stat.st_ctime))
            
            # Count file types
            ext = Path(filename).suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
            
            # Check if analyzed
            analysis_filename = f"{Path(filename).stem}_analysis.json"
            analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
            if os.path.exists(analysis_filepath) or filename in analysis_results:
                analyzed_count += 1
    
    file_stats.sort(key=lambda x: x[1])
    
    return StatsResponse(
        total_files=total_files,
        total_size=upload_stats["total_size"],
        upload_folder_size=get_folder_size(UPLOAD_FOLDER),
        analyzed_files=analyzed_count,
        pending_analysis=upload_stats["pending_analysis"],
        oldest_file=file_stats[0][0] if file_stats else None,
        newest_file=file_stats[-1][0] if file_stats else None,
        file_types=file_types
    )

@app.get("/analyses")
async def get_all_analyses(
    limit: int = Query(50, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    x_api_key: str = Header(None)
):
    """Get all analysis results"""
    verify_api_key(x_api_key)
    
    master_file = os.path.join(ANALYSIS_FOLDER, "all_analyses.json")
    if not os.path.exists(master_file):
        return {"status": "success", "analyses": [], "total": 0}
    
    try:
        with open(master_file, "r") as f:
            all_analyses = json.load(f)
        
        # Apply pagination
        paginated_results = all_analyses[offset:offset + limit]
        
        return {
            "status": "success",
            "analyses": paginated_results,
            "total": len(all_analyses),
            "offset": offset,
            "limit": limit
        }
    except Exception as e:
        logger.error(f"❌ Error reading analyses: {e}")
        raise HTTPException(status_code=500, detail="Error reading analyses")

# Keep existing endpoints (download, delete, cleanup, logs, system-info, test-upload)
@app.get("/files/{filename}")
async def download_file(
    filename: str = FastAPIPath(...),
    x_api_key: str = Header(None)
):
    """Download a specific file"""
    verify_api_key(x_api_key)
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        filepath,
        media_type=mimetypes.guess_type(filepath)[0] or "application/octet-stream",
        filename=filename
    )

@app.delete("/files/{filename}")
async def delete_file(
    filename: str = FastAPIPath(...),
    x_api_key: str = Header(None)
):
    """Delete a specific file and its analysis"""
    verify_api_key(x_api_key)
    
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_size = os.path.getsize(filepath)
        os.remove(filepath)
        
        # Remove analysis files
        analysis_filename = f"{Path(filename).stem}_analysis.json"
        analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
        if os.path.exists(analysis_filepath):
            os.remove(analysis_filepath)
        
        # Remove from memory
        if filename in analysis_results:
            del analysis_results[filename]
        
        # Update stats
        upload_stats["total_uploads"] = max(0, upload_stats["total_uploads"] - 1)
        upload_stats["total_size"] = max(0, upload_stats["total_size"] - file_size)
        
        logger.info(f"🗑️ Deleted file and analysis: {filename}")
        return {"status": "success", "message": f"File {filename} and its analysis deleted"}
        
    except Exception as e:
        logger.error(f"❌ Error deleting file: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {"message": "pong", "timestamp": datetime.now().isoformat()}

# -------------------------
# 🌍 Run FastAPI + Ngrok
# -------------------------
def start_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

def main():
    logger.info("🚦 Starting Enhanced FastAPI server with AI Image Analysis...")

    # Start FastAPI server in a thread
    server_thread = threading.Thread(target=start_fastapi, daemon=True)
    server_thread.start()

    # Wait a bit to make sure server starts
    time.sleep(3)

    # Start ngrok tunnel
    public_url = ngrok.connect(PORT, bind_tls=False)
    logger.info(f"🌐 Public URL: {public_url}")
    logger.info(f"📡 API Endpoints:")
    logger.info(f"   • Health Check: {public_url}/health")
    logger.info(f"   • Upload & Analyze: {public_url}/upload")
    logger.info(f"   • Get Analysis: {public_url}/analyze/{{filename}}")
    logger.info(f"   • List Files: {public_url}/files")
    logger.info(f"   • All Analyses: {public_url}/analyses")
    logger.info(f"   • Stats: {public_url}/stats")
    logger.info(f"   • Docs: {public_url}/docs")
    logger.info(f"🔐 Use X-API-Key: {API_KEY}")
    logger.info(f"📂 Upload destination: {os.path.abspath(UPLOAD_FOLDER)}")
    logger.info(f"🧠 Analysis results: {os.path.abspath(ANALYSIS_FOLDER)}")
    logger.info(f"🤖 AI Model: {AI_MODEL}")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down...")
        ngrok.disconnect(public_url)
        ngrok.kill()

if __name__ == "__main__":
    main()
