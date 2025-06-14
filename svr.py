import os
import time
import logging
import json
import uuid
import base64
import requests
import concurrent.futures
import re
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
MAX_RESPONSE_SIZE = 15 * 1024 * 1024  # 15MB max response size
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff'}
RATE_LIMIT = 60  # requests per minute

# AI Analysis API Configuration
AI_API_URL = "https://text.pollinations.ai/openai"
AI_MODEL = "o4-mini"
AI_MAX_TOKENS = 5500
AI_RETRIES = 3  # Reduced from 5 to 3 for faster failure detection
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
    failed_analysis: int
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
    description="A comprehensive image upload, analysis, and management server with robust error handling",
    version="3.1.0"
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
    "failed_analysis": 0,
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
    try:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        logger.error(f"❌ Error generating hash for {filepath}: {e}")
        return "hash_error"

def get_folder_size(folder_path: str) -> int:
    """Calculate total size of folder"""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(folder_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
    except Exception as e:
        logger.error(f"❌ Error calculating folder size for {folder_path}: {e}")
    return total_size

def is_allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"❌ Error encoding image to base64: {e}")
        raise

def clean_for_json_serialization(data, max_depth=10):
    """Recursively clean data to ensure it can be JSON serialized"""
    if max_depth <= 0:
        return "MAX_DEPTH_REACHED"
    
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            try:
                # Ensure key is string and clean
                clean_key = str(key)[:200]  # Limit key length
                cleaned[clean_key] = clean_for_json_serialization(value, max_depth - 1)
            except Exception as e:
                logger.warning(f"⚠️ Skipping problematic dict key: {key} - {e}")
                cleaned[f"error_key_{len(cleaned)}"] = "SERIALIZATION_ERROR"
        return cleaned
    
    elif isinstance(data, list):
        cleaned = []
        for i, item in enumerate(data):
            try:
                cleaned.append(clean_for_json_serialization(item, max_depth - 1))
            except Exception as e:
                logger.warning(f"⚠️ Skipping problematic list item at index {i}: {e}")
                cleaned.append("SERIALIZATION_ERROR")
        return cleaned
    
    elif isinstance(data, str):
        try:
            # Limit string length to prevent massive responses
            if len(data) > 50000:  # 50KB limit per string
                data = data[:50000] + "... [TRUNCATED]"
            
            # Remove control characters except newlines, tabs, and carriage returns
            cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', data)
            
            # Ensure it's valid UTF-8
            cleaned = cleaned.encode('utf-8', errors='ignore').decode('utf-8')
            
            return cleaned
        except Exception as e:
            logger.warning(f"⚠️ String cleaning failed: {e}")
            return "STRING_SERIALIZATION_ERROR"
    
    elif isinstance(data, (int, float, bool)) or data is None:
        # Handle potential infinity or NaN values
        if isinstance(data, float):
            if not (data == data):  # NaN check
                return "NaN"
            elif data == float('inf'):
                return "Infinity"
            elif data == float('-inf'):
                return "-Infinity"
        return data
    
    else:
        # Convert other types to string with length limit
        try:
            str_repr = str(data)
            if len(str_repr) > 1000:
                str_repr = str_repr[:1000] + "... [TRUNCATED]"
            return str_repr
        except Exception:
            return "UNKNOWN_TYPE_SERIALIZATION_ERROR"

def validate_json_structure(data):
    """Validate that data can be serialized to JSON"""
    try:
        json.dumps(data, ensure_ascii=False)
        return True, None
    except Exception as e:
        return False, str(e)

def analyze_image_with_ai(image_path: str, filename: str) -> Dict[str, Any]:
    """Send image to AI API for analysis with robust error handling"""
    try:
        encoded_image = encode_image_to_base64(image_path)
    except Exception as e:
        logger.error(f"❌ Failed to encode image {filename}: {e}")
        return {
            "status": "error",
            "error": f"Image encoding failed: {str(e)}",
            "analysis_time": datetime.now().isoformat()
        }

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
                            "Conduct a comprehensive visual analysis of the image in detail, with a word count between "
                            "**2000 to 3500 words**. Maintain an academic and formal tone. "
                            "Focus on completeness, accuracy, and specificity.\n\n"

                            "**IMPORTANT: Your response must be valid JSON-compatible text. Avoid special characters that might break JSON parsing.**\n\n"

                            "**KEY DIRECTIVES:**\n"
                            "- Focus solely on what **is visible** in the image — do **NOT** speculate about what is absent.\n"
                            "- Provide relevant **scientific explanations** for observed phenomena when applicable.\n"
                            "- Identify the image's **theme, artistic intention, or communicative objective**.\n\n"

                            "**ANALYZE THESE CATEGORIES IN DETAIL:**\n\n"

                            "1) **Visible Objects and Spatial Relationships**: List and count all objects with their positions.\n"
                            "2) **People** (if present): Describe postures, clothing, apparent age, expressions.\n"
                            "3) **Actions and Interactions**: Explain what is happening in the image.\n"
                            "4) **Background Elements**: Describe environmental context.\n"
                            "5) **Lighting and Shadows**: Analyze light sources and their effects (skip for text-only images).\n"
                            "6) **Colors and Contrast**: Describe color schemes and visual impact.\n"
                            "7) **Textures and Materials**: Detail surface qualities (skip for text-only images).\n"
                            "8) **Camera Perspective**: Discuss angle, framing, and composition.\n"
                            "9) **Artistic Style**: Identify visual style and symbolic elements.\n"
                            "10) **Visible Text**: Transcribe ALL visible text exactly as shown, with translations if needed.\n"
                            "11) **Cultural Context**: Infer cultural/historical elements from visual cues (skip for text-only images).\n"
                            "12) **Emotional Tone**: Assess the mood and atmosphere with visual evidence.\n"
                            "13) **Current Action**: Describe the ongoing moment or event.\n"
                            "14) **Scientific Concepts**: Explain any scientific diagrams or processes shown (if applicable).\n"
                            "15) **Humor/Memes**: Identify and explain any humorous content (if applicable).\n"
                            "16) **Data Visualization**: Interpret charts, graphs, or statistical content (if applicable).\n"
                            "17) **Species Information**: Provide scientific names and characteristics for any animals/plants (if applicable).\n\n"

                            "**Format your response as clear, structured text without special formatting characters that might break JSON.**"
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
                timeout=120  # Increased timeout
            )
            
            # Check response size
            if len(response.content) > MAX_RESPONSE_SIZE:
                logger.warning(f"⚠️ Response too large for {filename}: {len(response.content)} bytes")
                return {
                    "status": "error",
                    "error": f"AI response too large: {len(response.content)} bytes",
                    "analysis_time": datetime.now().isoformat()
                }
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    logger.info(f"✅ AI analysis completed for: {filename} on attempt {attempt + 1}")
                    
                    # Clean the result to ensure JSON serialization works
                    cleaned_result = clean_for_json_serialization(result)
                    
                    # Validate the cleaned result
                    is_valid, error_msg = validate_json_structure(cleaned_result)
                    if not is_valid:
                        logger.warning(f"⚠️ Cleaned result still not JSON serializable for {filename}: {error_msg}")
                        # Return a simplified version
                        return {
                            "status": "partial_success",
                            "analysis": {
                                "message": "Analysis completed but content was too complex for full serialization",
                                "summary": str(cleaned_result)[:2000] + "... [TRUNCATED]" if len(str(cleaned_result)) > 2000 else str(cleaned_result),
                                "original_error": error_msg
                            },
                            "analysis_time": datetime.now().isoformat(),
                            "model_used": AI_MODEL
                        }
                    
                    return {
                        "status": "success",
                        "analysis": cleaned_result,
                        "analysis_time": datetime.now().isoformat(),
                        "model_used": AI_MODEL
                    }
                    
                except json.JSONDecodeError as json_error:
                    logger.error(f"❌ AI API returned invalid JSON for {filename}: {json_error}")
                    logger.debug(f"Response preview: {response.text[:500]}...")
                    
                    # Try to extract useful content from malformed JSON
                    try:
                        # Look for content between quotes or brackets
                        content_match = re.search(r'"content":\s*"([^"]*)"', response.text)
                        if content_match:
                            extracted_content = content_match.group(1)
                            return {
                                "status": "partial_success",
                                "analysis": {
                                    "message": "AI response was malformed JSON, extracted content",
                                    "content": extracted_content[:3000] + "... [TRUNCATED]" if len(extracted_content) > 3000 else extracted_content
                                },
                                "analysis_time": datetime.now().isoformat(),
                                "model_used": AI_MODEL,
                                "note": "Original response had JSON formatting issues"
                            }
                    except Exception:
                        pass
                    
                    return {
                        "status": "error",
                        "error": f"AI API returned invalid JSON: {str(json_error)}",
                        "analysis_time": datetime.now().isoformat(),
                        "raw_response_preview": response.text[:500]
                    }
                    
            elif response.status_code in [429, 500, 502, 503, 504]:  # Retryable errors
                if attempt < AI_RETRIES - 1:
                    wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                    logger.warning(f"⚠️ AI API error for {filename}: {response.status_code}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"❌ AI API error for {filename} after {AI_RETRIES} attempts: {response.status_code}")
                    return {
                        "status": "error",
                        "error": f"API Error {response.status_code} after {AI_RETRIES} attempts",
                        "analysis_time": datetime.now().isoformat()
                    }
            else:
                logger.error(f"❌ Non-retryable AI API error for {filename}: {response.status_code}")
                return {
                    "status": "error",
                    "error": f"API Error {response.status_code}: {response.text[:300]}",
                    "analysis_time": datetime.now().isoformat()
                }
                
        except requests.exceptions.Timeout:
            if attempt < AI_RETRIES - 1:
                wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(f"⏰ AI API request timed out for {filename}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue
            else:
                logger.error(f"❌ AI API request timed out for {filename} after {AI_RETRIES} attempts.")
                return {
                    "status": "error",
                    "error": "AI API request timed out after multiple attempts",
                    "analysis_time": datetime.now().isoformat()
                }
                
        except requests.exceptions.RequestException as e:
            if attempt < AI_RETRIES - 1:
                wait_time = AI_RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(f"🌐 Network error during AI analysis for {filename}: {str(e)}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                continue
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
                "error": f"Unexpected error: {str(e)}",
                "analysis_time": datetime.now().isoformat()
            }
    
    return {
        "status": "error",
        "error": f"Failed to get AI analysis after {AI_RETRIES} attempts",
        "analysis_time": datetime.now().isoformat()
    }

def save_analysis_to_json(filename: str, analysis_result: Dict[str, Any], file_info: Dict[str, Any]):
    """Save analysis result and file info to JSON with robust error handling"""
    try:
        analysis_data = {
            "filename": filename,
            "file_info": file_info,
            "analysis_result": analysis_result,
            "saved_time": datetime.now().isoformat()
        }
        
        # Validate the data before saving
        is_valid, error_msg = validate_json_structure(analysis_data)
        if not is_valid:
            logger.warning(f"⚠️ Analysis data not JSON serializable for {filename}: {error_msg}")
            # Create a simplified version
            analysis_data = {
                "filename": filename,
                "file_info": {
                    "filename": file_info.get("filename", "unknown"),
                    "size": file_info.get("size", 0),
                    "upload_time": file_info.get("upload_time", ""),
                    "mime_type": file_info.get("mime_type", "unknown")
                },
                "analysis_result": {
                    "status": analysis_result.get("status", "error"),
                    "error": "Analysis content was too complex for JSON serialization",
                    "analysis_time": analysis_result.get("analysis_time", datetime.now().isoformat()),
                    "model_used": analysis_result.get("model_used", "unknown"),
                    "simplified_content": str(analysis_result)[:1000] + "..." if len(str(analysis_result)) > 1000 else str(analysis_result)
                },
                "saved_time": datetime.now().isoformat(),
                "note": "Original analysis was simplified due to JSON serialization issues"
            }
        
        # Save individual analysis file
        analysis_filename = f"{Path(filename).stem}_analysis.json"
        analysis_filepath = os.path.join(ANALYSIS_FOLDER, analysis_filename)
        
        with open(analysis_filepath, "w", encoding='utf-8') as f:
            json.dump(analysis_data, f, indent=2, ensure_ascii=False)
        
        # Update master analysis file with safer handling
        master_file = os.path.join(ANALYSIS_FOLDER, "all_analyses.json")
        
        # Load existing analyses
        all_analyses = []
        if os.path.exists(master_file):
            try:
                with open(master_file, "r", encoding='utf-8') as f:
                    all_analyses = json.load(f)
                if not isinstance(all_analyses, list):
                    logger.warning("⚠️ Master analysis file contains non-list data, resetting...")
                    all_analyses = []
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Master analysis file corrupted, creating backup: {e}")
                # Backup the corrupted file
                backup_file = f"{master_file}.backup_{int(time.time())}"
                try:
                    os.rename(master_file, backup_file)
                    logger.info(f"📦 Corrupted master file backed up to: {backup_file}")
                except Exception as backup_error:
                    logger.error(f"❌ Failed to backup corrupted file: {backup_error}")
                all_analyses = []
            except Exception as e:
                logger.error(f"❌ Error reading master analysis file: {e}")
                all_analyses = []
        
        # Add new analysis (limit total entries to prevent file from growing too large)
        all_analyses.append(analysis_data)
        if len(all_analyses) > 1000:  # Keep only last 1000 analyses
            all_analyses = all_analyses[-1000:]
            logger.info("🔄 Master analysis file pruned to last 1000 entries")
        
        # Save updated master file
        with open(master_file, "w", encoding='utf-8') as f:
            json.dump(all_analyses, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Analysis saved to JSON for: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error saving analysis to JSON for {filename}: {str(e)}")
        
        # Last resort: save a minimal error record
        try:
            error_data = {
                "filename": filename,
                "error": f"Failed to save analysis: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "file_size": file_info.get("size", 0) if isinstance(file_info, dict) else 0
            }
            
            error_filename = f"{Path(filename).stem}_error.json"
            error_filepath = os.path.join(ANALYSIS_FOLDER, error_filename)
            
            with open(error_filepath, "w", encoding='utf-8') as f:
                json.dump(error_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Error record saved for: {filename}")
            
        except Exception as final_error:
            logger.error(f"❌ Even error record save failed for {filename}: {str(final_error)}")

def process_image_analysis(filepath: str, filename: str, file_info: Dict[str, Any]):
    """Background task to analyze image with comprehensive error handling"""
    try:
        upload_stats["pending_analysis"] += 1
        logger.info(f"🎯 Starting analysis pipeline for: {filename}")
        
        # Verify file still exists
        if not os.path.exists(filepath):
            logger.error(f"❌ File not found for analysis: {filepath}")
            upload_stats["pending_analysis"] = max(0, upload_stats["pending_analysis"] - 1)
            upload_stats["failed_analysis"] += 1
            return
        
        # Perform AI analysis
        analysis_result = analyze_image_with_ai(filepath, filename)
        
        # Store in memory (with size limit)
        if len(analysis_results) > 500:  # Limit memory usage
            # Remove oldest entries
            oldest_keys = list(analysis_results.keys())[:100]
            for key in oldest_keys:
                del analysis_results[key]
            logger.info("🔄 Analysis results cache pruned")
        
        analysis_results[filename] = analysis_result
        
        # Save to JSON
        save_analysis_to_json(filename, analysis_result, file_info)
        
        # Update stats
        upload_stats["pending_analysis"] = max(0, upload_stats["pending_analysis"] - 1)
        
        if analysis_result["status"] == "success":
            upload_stats["analyzed_files"] += 1
            logger.info(f"✅ Analysis pipeline completed successfully for: {filename}")
        elif analysis_result["status"] == "partial_success":
            upload_stats["analyzed_files"] += 1
            logger.info(f"⚠️ Analysis pipeline completed with partial success for: {filename}")
        else:
            upload_stats["failed_analysis"] += 1
            logger.warning(f"⚠️ Analysis pipeline failed for: {filename}")
        
    except Exception as e:
        logger.error(f"❌ Error in analysis pipeline for {filename}: {str(e)}")
        upload_stats["pending_analysis"] = max(0, upload_stats["pending_analysis"] - 1)
        upload_stats["failed_analysis"] += 1

# -------------------------
# 📡 API Endpoints
# -------------------------

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Enhanced Image Server with AI Analysis",
        "version": "3.1.0",
        "features": ["Upload", "AI Analysis", "File Management", "Statistics", "Robust Error Handling"],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "upload": "/upload",
            "files": "/files",
            "stats": "/stats",
            "analyses": "/analyses"
        },
        "status": "operational"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        uptime = time.time() - start_time
        uptime_str = str(timedelta(seconds=int(uptime)))
        
        # Disk usage
        disk_usage = psutil.disk_usage(os.getcwd())
        disk_info = {
            "total": disk_usage.total,
            "used": disk_usage.used,
            "free": disk_usage.free,
            "percent": round((disk_usage.used / disk_usage.total) * 100, 2)
        }
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_info = {
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
            "percent": round(memory.percent, 2)
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime=uptime_str,
            version="3.1.0",
            disk_usage=disk_info,
            memory_usage=memory_info
        )
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

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
        raise HTTPException(status_code=400, detail=f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        logger.error(f"❌ Error reading uploaded file: {e}")
        raise HTTPException(status_code=400, detail="Error reading file content")
    
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file not allowed")
    
    # Generate unique filename
    file_id = str(uuid.uuid4())
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    extension = Path(file.filename).suffix
    safe_original_name = re.sub(r'[^\w\-_\.]', '_', file.filename)  # Sanitize filename
    filename = f"{timestamp}_{file_id}_{
