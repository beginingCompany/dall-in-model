import uvicorn
import webbrowser
import time
import threading
import logging
import sys
import os
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("server")

def check_port_available(port):
    """Check if the port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('127.0.0.1', port))
            return True
        except socket.error:
            return False

def open_browser():
    """Open browser after server starts."""
    try:
        # Wait a bit for the server to start
        time.sleep(2)
        # Open the browser with the application URL
        url = f"http://127.0.0.1:8000"
        webbrowser.open(url)
        logger.info(f"Opened browser at {url}")
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    port = start_port
    attempts = 0
    
    while attempts < max_attempts:
        if check_port_available(port):
            return port
        port += 1
        attempts += 1
    
    logger.warning(f"Could not find an available port after {max_attempts} attempts.")
    return start_port  # Return the original port and let uvicorn handle any errors

if __name__ == "__main__":
    try:
        # Find an available port
        port = find_available_port()
        logger.info(f"Using port {port}")
        
        # Check for required Python packages
        try:
            import fastapi
            import pydantic
            logger.info("Required packages are installed.")
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.info("Please install required packages with: pip install -r requirements.txt")
            sys.exit(1)
        
        # Display helpful information
        logger.info("=" * 60)
        logger.info("         PERSONALITY ANALYSIS SYSTEM")
        logger.info("=" * 60)
        logger.info("For effective personality analysis, provide:")
        logger.info("1. A detailed self-description (3-4 sentences minimum)")
        logger.info("2. Include emotional, social, cognitive, and behavioral traits")
        logger.info("3. Add follow-up Q&A pairs for more depth")
        logger.info("=" * 60)
        logger.info("TROUBLESHOOTING:")
        logger.info("- If you see 'minimal_input' warning, add more details")
        logger.info("- Use debug mode with ?debug=true appended to URL")
        logger.info("- Try the examples: python examples/effective_input_example.py")
        logger.info("=" * 60)
        
        # Start the browser opening in a separate thread
        threading.Thread(target=open_browser).start()
        
        # Log startup information
        logger.info("Starting API server...")
        logger.info(f"Access the UI at http://127.0.0.1:{port}")
        
        # Start the FastAPI server with additional logging
        uvicorn.run(
            "app.api:app", 
            host="127.0.0.1", 
            port=port, 
            reload=True,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
