import os
import time
import json
import logging
import requests
from datetime import datetime, timedelta
from pathlib import Path
import sys
import urllib.parse

# --- 🔧 CONFIGURATION ---
# Local folder where new images appear
CAMERA_FOLDER = "/home/humair/Downloads"
# Cache files to track uploaded and analyzed images
UPLOAD_CACHE_FILE = "/home/humair/_ai_/rand/upload_cache.json"
ANALYSIS_CACHE_FILE = "/home/humair/_ai_/rand/analysis_cache.json"

# FastAPI server details
BASE_URL = "http://localhost:5000" # Your Ngrok base URL
UPLOAD_ENDPOINT = "/upload"  # Endpoint for image uploads
ANALYSIS_ENDPOINT_PREFIX = "/analyze/" # Endpoint prefix for analysis results
STATS_ENDPOINT = "/stats" # Endpoint for server statistics
API_KEY = "mysecurekey123"

# Operational settings
CHECK_INTERVAL = 15  # Seconds between scanning the camera folder
RETRY_LIMIT = 3  # Max retries for failed network requests
REQUEST_TIMEOUT = 30  # Seconds before a request times out
UPLOAD_DELAY = 2  # Seconds to wait between consecutive image uploads
ENABLE_AUTO_ANALYSIS = True  # Whether to request AI analysis after upload
ANALYSIS_CHECK_DELAY = 20  # Seconds to wait before checking analysis results (not used directly in loop, but good for context)
MAX_CACHE_AGE = 7 * 24 * 3600 *29000000000000000000 # Cache entries older than 7 days are considered stale (in seconds)

# --- 🪵 LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
    handlers=[
        logging.FileHandler("mobile_uploader.log"),
        logging.StreamHandler(sys.stdout) # Ensure logs go to console
    ]
)


def get_logger(name):
    """Helper to get a logger instance."""
    return logging.getLogger(name)
logger = get_logger(__name__)

# --- 📁 HELPER FUNCTIONS ---

def validate_config():
    """
    Validates essential configuration settings at startup.
    Exits the script if critical settings are invalid or the server is unreachable.
    """
    if not Path(CAMERA_FOLDER).is_dir():
        logger.critical(f"❌ Camera folder does not exist or is not a directory: {CAMERA_FOLDER}")
        sys.exit(1)
    
    parsed_url = urllib.parse.urlparse(BASE_URL)
    if not parsed_url.scheme or not parsed_url.netloc:
        logger.critical(f"❌ Invalid BASE_URL. Please ensure it starts with http:// or https:// and has a valid domain: {BASE_URL}")
        sys.exit(1)
    
    if not API_KEY:
        logger.critical("❌ API_KEY is not set. Please provide a valid API key in the configuration.")
        sys.exit(1)
    
    # Test server connectivity
    try:
        ping_url = urllib.parse.urljoin(BASE_URL, "/ping") # Use the /ping endpoint for a lightweight check
        response = requests.get(ping_url, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200 and response.json().get('message') == 'pong':
            logger.info(f"✅ Server reachable at {BASE_URL}. Ping successful.")
        else:
            logger.warning(f"⚠️ Server returned unexpected status {response.status_code} or response during ping.")
            logger.warning(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        logger.critical(f"❌ Cannot connect to server at {BASE_URL}: {e}. Please check the server address and network connection.")
        sys.exit(1)

def load_cache(cache_file: str) -> dict:
    """Loads cache data from a specified JSON file."""
    if Path(cache_file).exists():
        try:
            with open(cache_file, "r") as f:
                cache = json.load(f)
            logger.debug(f"💾 Loaded cache from {cache_file}")
            return cache
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️ Failed to parse JSON from cache file {cache_file}: {e}. Starting with an empty cache.")
            return {}
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cache {cache_file}: {e}. Starting with an empty cache.")
            return {}
    return {}

def save_cache(cache_data: dict, cache_file: str):
    """Saves cache data to a specified JSON file."""
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
        logger.debug(f"💾 Cache saved to {cache_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save cache {cache_file}: {e}")

def clean_cache(cache_data: dict, cache_file: str, max_age: int = MAX_CACHE_AGE) -> dict:
    """
    Removes stale cache entries older than `max_age` seconds.
    `upload_time` in cache entries is expected to be in ISO format.
    """
    now = time.time()
    cleaned_cache = {}
    removed_count = 0

    for filename, data in cache_data.items():
        upload_time_str = data.get("upload_time")
        if upload_time_str:
            try:
                upload_dt = datetime.fromisoformat(upload_time_str)
                age = now - upload_dt.timestamp()
                if age <= max_age:
                    cleaned_cache[filename] = data
                else:
                    logger.info(f"🗑️ Removed stale cache entry for {filename} (age: {timedelta(seconds=int(age))})")
                    removed_count += 1
            except ValueError:
                logger.warning(f"⚠️ Invalid timestamp format in cache for {filename}, keeping entry.")
                cleaned_cache[filename] = data
        else:
            # If no upload_time, assume it's valid or handle as needed
            cleaned_cache[filename] = data
    
    if removed_count > 0:
        logger.info(f"🧹 Cleaned {removed_count} stale entries from {cache_file}.")
        save_cache(cleaned_cache, cache_file) # Save after cleaning if changes were made
    return cleaned_cache

def get_all_images() -> list:
    """
    Retrieves all image files from the CAMERA_FOLDER,
    sorted by modification time (newest first).
    """
    try:
        images = []
        for f in os.listdir(CAMERA_FOLDER):
            filepath = Path(CAMERA_FOLDER) / f
            if filepath.is_file() and f.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff")):
                images.append(f)
        
        # Sort by modification time, newest first
        images.sort(key=lambda x: (Path(CAMERA_FOLDER) / x).stat().st_mtime, reverse=True)
        return images
    except Exception as e:
        logger.error(f"❌ Error scanning camera folder {CAMERA_FOLDER}: {e}")
        return []

def get_file_info(filepath: Path) -> dict:
    """Extracts basic information about a file."""
    try:
        stat = filepath.stat()
        return {
            "filename": filepath.name,
            "size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "full_path": str(filepath)
        }
    except FileNotFoundError:
        logger.error(f"❌ File not found when getting info: {filepath}")
        return None
    except Exception as e:
        logger.error(f"❌ Error getting file info for {filepath}: {e}")
        return None

def upload_image(filepath: Path, original_filename: str) -> dict:
    """
    Uploads an image to the FastAPI server with retry logic.
    Returns a dictionary with upload status and server response details.
    """
    file_info = get_file_info(filepath)
    if not file_info:
        return {"status": "failed", "message": f"Could not get file info for {original_filename}"}
    
    upload_url = urllib.parse.urljoin(BASE_URL, UPLOAD_ENDPOINT)
    logger.info(f"📤 Uploading: {original_filename} ({file_info['size']} bytes) to {upload_url}")
    
    for attempt in range(RETRY_LIMIT):
        try:
            with open(filepath, "rb") as img_file:
                files = {"file": (original_filename, img_file, "image/jpeg")}
                headers = {"X-API-Key": API_KEY}
                params = {"analyze": "true" if ENABLE_AUTO_ANALYSIS else "false"}
                
                response = requests.post(
                    upload_url,
                    files=files,
                    headers=headers,
                    params=params,
                    timeout=REQUEST_TIMEOUT
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Upload successful for {original_filename}.")
                    logger.info(f"   Server Filename: {result.get('filename', 'N/A')}")
                    logger.info(f"   File ID: {result.get('file_id', 'N/A')}")
                    logger.info(f"   Analysis Status: {result.get('analysis_status', 'N/A')}")
                    
                    return {
                        "status": "success",
                        "upload_time": datetime.now().isoformat(),
                        "server_response": result,
                        "file_info": file_info,
                        "server_filename": result.get('filename'),
                        "file_id": result.get('file_id'),
                        "analysis_status": result.get('analysis_status', 'unknown')
                    }
                elif response.status_code == 403:
                    logger.error(f"❌ API Key Invalid (403) for {original_filename}. Check your API_KEY.")
                    break # Do not retry on invalid API key
                elif response.status_code == 405:
                    logger.error(f"❌ Method Not Allowed (405) at {upload_url}. Verify that the server endpoint supports POST requests.")
                    break # Do not retry, this is a configuration issue
                else:
                    logger.warning(f"⚠️ Upload failed for {original_filename} (Attempt {attempt + 1}/{RETRY_LIMIT}). Status: {response.status_code}, Response: {response.text}")
                    
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Upload timeout for {original_filename} (Attempt {attempt + 1}/{RETRY_LIMIT}).")
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 Connection error for {original_filename} (Attempt {attempt + 1}/{RETRY_LIMIT}). Check server and network.")
        except Exception as e:
            logger.error(f"❌ Unexpected error during upload for {original_filename} (Attempt {attempt + 1}/{RETRY_LIMIT}): {e}")
        
        if attempt < RETRY_LIMIT - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.info(f"⏳ Waiting {wait_time}s before retrying upload...")
            time.sleep(wait_time)
        
    logger.error(f"❌ Upload failed permanently after {RETRY_LIMIT} attempts for {original_filename}.")
    return {"status": "failed", "message": "Upload failed after multiple retries."}

def check_analysis_result(server_filename: str) -> dict:
    """
    Checks the AI analysis result for a given server-side filename.
    Returns the analysis content and status, or None if not found/error.
    """
    if not server_filename:
        logger.warning("No server_filename provided to check analysis.")
        return {"status": "error", "message": "No server filename."}
    
    analysis_url = urllib.parse.urljoin(BASE_URL, f"{ANALYSIS_ENDPOINT_PREFIX}{server_filename}")
    
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(analysis_url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"🧠 Analysis retrieved for: {server_filename}")
            
            analysis_content = result.get('analysis', {})
            content = ""
            # Extract content from the nested AI response structure
            if isinstance(analysis_content, dict) and 'analysis' in analysis_content:
                choices = analysis_content['analysis'].get('choices', [])
                if choices and len(choices) > 0:
                    content = choices[0].get('message', {}).get('content', '')
            else:
                content = str(analysis_content) # Fallback for unexpected structure
            
            if content:
                logger.info(f"📝 AI Analysis preview: {content[:150]}...") # Show a bit more of the preview
            
            return {
                "status": "completed", # Changed to 'completed' as we received the analysis
                "analysis_time": result.get('analysis_time'),
                "content": content,
                "full_response": result
            }
            
        elif response.status_code == 404:
            logger.info(f"⏳ Analysis not ready yet for: {server_filename}. Still pending.")
            return {"status": "pending"}
        else:
            logger.warning(f"⚠️ Analysis check failed for {server_filename}: Status {response.status_code} - {response.text}")
            return {"status": "error", "message": response.text}
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Network error checking analysis for {server_filename}: {e}")
        return {"status": "error", "message": f"Network error: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Unexpected error checking analysis for {server_filename}: {e}")
        return {"status": "error", "message": f"Unhandled error: {str(e)}"}

def get_server_stats():
    """Fetches and logs server statistics."""
    stats_url = urllib.parse.urljoin(BASE_URL, STATS_ENDPOINT)
    try:
        headers = {"X-API-Key": API_KEY}
        response = requests.get(stats_url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if response.status_code == 200:
            stats = response.json()
            logger.info("--- 📊 Server Stats ---")
            logger.info(f"   Total Files: {stats.get('total_files', 0)}")
            logger.info(f"   Analyzed Files: {stats.get('analyzed_files', 0)}")
            logger.info(f"   Pending Analysis: {stats.get('pending_analysis', 0)}")
            logger.info(f"   Upload Folder Size: {stats.get('upload_folder_size', 0) / (1024*1024):.2f} MB")
            logger.info("--------------------")
            return stats
        else:
            logger.warning(f"⚠️ Failed to retrieve server stats: Status {response.status_code} - {response.text}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Error connecting to server for stats: {e}")
    except Exception as e:
        logger.error(f"❌ Unexpected error getting server stats: {e}")
    return None

def process_pending_analyses(analysis_cache: dict):
    """Iterates through cached items and checks for completed analysis results."""
    pending_files = [
        filename for filename, data in analysis_cache.items()
        if data.get("analysis_status") in ["processing", "unknown", "pending"]
    ]
    
    if not pending_files:
        logger.info("✨ No pending analyses to check.")
        return
    
    logger.info(f"🔍 Checking status of {len(pending_files)} pending analyses...")
    
    for filename in pending_files:
        cache_entry = analysis_cache[filename]
        server_filename = cache_entry.get("server_filename")
        
        if server_filename:
            analysis_result = check_analysis_result(server_filename)
            
            if analysis_result and analysis_result.get("status") == "completed":
                cache_entry["analysis_result"] = analysis_result
                cache_entry["analysis_status"] = "completed"
                cache_entry["analysis_retrieved_time"] = datetime.now().isoformat()
                logger.info(f"✅ Analysis for {filename} is now COMPLETE.")
            elif analysis_result and analysis_result.get("status") == "error":
                cache_entry["analysis_status"] = "failed"
                cache_entry["analysis_error"] = analysis_result.get("message", "Unknown error")
                logger.error(f"❌ Analysis for {filename} FAILED: {cache_entry['analysis_error']}.")
            else:
                # Analysis is still pending, no update needed, or temporary error
                logger.debug(f"Analysis for {filename} still pending or encountered temporary issue.")

# --- 🔁 MAIN LOOP ---

def main():
    logger.info("--- 🚀 Starting Mobile Image Uploader with AI Analysis ---")
    logger.info(f"Monitoring: {CAMERA_FOLDER}")
    logger.info(f"Server URL: {BASE_URL}")
    logger.info(f"Auto Analysis: {'Enabled' if ENABLE_AUTO_ANALYSIS else 'Disabled'}")
    logger.info("--------------------------------------------------")
    
    # Perform initial configuration validation
    validate_config()
    
    # Load and clean caches
    upload_cache = load_cache(UPLOAD_CACHE_FILE)
    analysis_cache = load_cache(ANALYSIS_CACHE_FILE)
    upload_cache = clean_cache(upload_cache, UPLOAD_CACHE_FILE)
    analysis_cache = clean_cache(analysis_cache, ANALYSIS_CACHE_FILE)
    
    iteration_count = 0
    
    while True:
        try:
            iteration_count += 1
            logger.info(f"\n--- Scan #{iteration_count} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
            
            all_images = get_all_images()
            logger.info(f"Found {len(all_images)} images in {CAMERA_FOLDER}.")
            
            new_uploads = 0
            for img_name in all_images:
                # Check if image has already been uploaded successfully
                if img_name not in upload_cache or upload_cache[img_name].get("status") != "success":
                    filepath = Path(CAMERA_FOLDER) / img_name
                    if not filepath.exists(): # Should not happen if get_all_images is correct
                        logger.warning(f"Skipping {img_name} as it no longer exists at {filepath}.")
                        continue

                    logger.info(f"📸 New or unuploaded image detected: {img_name}")
                    
                    upload_result = upload_image(filepath, img_name)
                    
                    if upload_result and upload_result["status"] == "success":
                        upload_cache[img_name] = upload_result
                        save_cache(upload_cache, UPLOAD_CACHE_FILE)
                        
                        if ENABLE_AUTO_ANALYSIS:
                            # Initialize analysis cache entry if analysis was requested
                            analysis_cache[img_name] = {
                                "server_filename": upload_result.get("server_filename"),
                                "file_id": upload_result.get("file_id"),
                                "analysis_status": upload_result.get("analysis_status", "unknown"),
                                "upload_time": upload_result.get("upload_time"),
                                "analysis_requested": True
                            }
                            save_cache(analysis_cache, ANALYSIS_CACHE_FILE)
                        
                        new_uploads += 1
                        time.sleep(UPLOAD_DELAY)  # Pause to prevent server overload
                    else:
                        logger.error(f"❌ Failed to process upload for: {img_name}. Will retry later.")
                else:
                    logger.debug(f"Skipping already uploaded image: {img_name}")
            
            if new_uploads > 0:
                logger.info(f"🎉 Successfully processed {new_uploads} new uploads in this scan.")
            
            # Periodically check for pending analysis results
            if ENABLE_AUTO_ANALYSIS:
                process_pending_analyses(analysis_cache)
                save_cache(analysis_cache, ANALYSIS_CACHE_FILE) # Save after checking analyses
            
            # Display server stats periodically
            if iteration_count % 10 == 0: # Every 10 scans
                get_server_stats()
            
            logger.info(f"⏳ Waiting {CHECK_INTERVAL}s until the next scan...")
            time.sleep(CHECK_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("🛑 Uploader stopped by user (KeyboardInterrupt). Exiting.")
            break
        except Exception as e:
            logger.critical(f" catastrophic error in main loop: {e}", exc_info=True)
            logger.info("Attempting to recover in 10 seconds...")
            time.sleep(10)

def show_cached_analyses():
    """Displays all cached analysis results in a user-friendly format."""
    analysis_cache = load_cache(ANALYSIS_CACHE_FILE)
    
    if not analysis_cache:
        print("\n📭 No cached analyses found.")
        return
    
    print(f"\n📊 Displaying {len(analysis_cache)} Cached Analysis Items:")
    print("=" * 60)
    
    for filename, data in analysis_cache.items():
        print(f"\n📸 **Original File:** {filename}")
        print(f"   **Server Filename:** {data.get('server_filename', 'N/A')}")
        print(f"   **Upload Time:** {data.get('upload_time', 'N/A')}")
        print(f"   **Analysis Status:** {data.get('analysis_status', 'Unknown').upper()}")
        
        analysis_result = data.get('analysis_result')
        if analysis_result:
            content = analysis_result.get('content', 'No content available.')
            print("\n\n " , "+"*60 , "\n\n")
            print(f"   **Analysis Preview:** {content[:]}...") # Display a longer preview
            # Optionally, save full analysis to a text file
            # You could add logic here to write the full `content` to a .txt file next to the original image
        else:
            print("   **Analysis Content:** Not yet retrieved or failed.")
        print("-" * 40)
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "show-analyses":
        show_cached_analyses()
    else:
        try:
            main()
        except Exception as e:
            logger.critical(f"A fatal error occurred outside the main loop: {e}", exc_info=True)
            logger.info("Please review the logs for more details and ensure all configurations are correct.")
