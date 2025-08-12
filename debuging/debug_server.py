"""
Enhanced Server Runner - Start the API server with debug mode enabled

This script:
1. Sets up debugging output
2. Checks if required packages are installed
3. Starts the FastAPI server with better error handling
4. Opens a browser to the UI interface

Run with: python debug_server.py
"""
import os
import sys
import webbrowser
import time
import threading
import subprocess
import importlib.util
import logging
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("debug_server")

def check_imports(packages: List[str]) -> List[str]:
    """Check if required packages are installed"""
    missing_packages = []
    for package in packages:
        # Skip package check if it's in a comment
        if package.startswith('#'):
            continue
            
        # Handle version specifiers by taking just the package name
        pkg_name = package.split('==')[0].split('>=')[0].split('<=')[0].strip()
        
        try:
            importlib.util.find_spec(pkg_name)
        except ImportError:
            missing_packages.append(pkg_name)
    
    return missing_packages

def install_missing_packages(packages: List[str]):
    """Install missing packages"""
    if not packages:
        return
        
    logger.info(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + packages)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install packages: {e}")
        sys.exit(1)

def open_browser():
    """Open browser to the UI interface"""
    try:
        # Wait a bit for the server to start
        time.sleep(2)
        # Open with debug flag
        url = "http://127.0.0.1:8000/static/index.html?debug=true"
        webbrowser.open(url)
        logger.info(f"Opened browser at {url}")
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

def main():
    """Main function to run the server"""
    logger.info("Starting debug server")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    
    # Check required packages from requirements.txt
    requirements_file = os.path.join(project_root, "requirements.txt")
    required_packages = []
    
    if os.path.exists(requirements_file):
        logger.info(f"Reading requirements from {requirements_file}")
        with open(requirements_file, 'r') as f:
            required_packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Add additional packages needed for debugging
    debug_packages = ["uvicorn", "fastapi"]
    required_packages.extend(debug_packages)
    
    # Check if packages are installed
    missing_packages = check_imports([pkg.split('==')[0].split('>=')[0].split('<=')[0] for pkg in required_packages])
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        install_missing_packages(missing_packages)
    
    # Start the browser opening in a separate thread
    threading.Thread(target=open_browser).start()
    
    # Start the server with more debugging output
    logger.info("Starting the FastAPI server...")
    os.environ['PYTHONPATH'] = project_root
    
    try:
        import uvicorn
        uvicorn.run(
            "app.api:app",
            host="127.0.0.1", 
            port=8000, 
            reload=False,  # We're handling debug mode directly
            log_level="debug"
        )
    except ImportError:
        logger.error("Failed to import uvicorn. Make sure it's installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
