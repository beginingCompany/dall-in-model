"""
Server Starter with Log Explanation
This script starts the API server and explains common log messages
"""
import os
import sys
import uvicorn
import time
import threading
import webbrowser

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    """Open browser after server starts"""
    time.sleep(2)  # Wait for server to start
    url = "http://127.0.0.1:8000/static/index.html"
    print(f"\nOpening browser at {url}")
    webbrowser.open(url)

def print_log_guide():
    """Print a guide to common log messages"""
    print("\n" + "="*60)
    print("COMMON LOG MESSAGES EXPLAINED")
    print("="*60)
    
    print("\n1. INFO: 127.0.0.1:XXXXX - \"HEAD / HTTP/1.1\" 405 Method Not Allowed")
    print("   ✓ FIXED: This was a server issue with HEAD requests")
    print("     This message appeared when checking server connectivity")
    
    print("\n2. INFO: 127.0.0.1:XXXXX - \"POST /analyze-personality HTTP/1.1\" 200 OK")
    print("   ✓ NORMAL: Successful request to the personality analyzer")
    
    print("\n3. INFO: 127.0.0.1:XXXXX - \"POST /analyze-personality HTTP/1.1\" 500 Internal Server Error")
    print("   ⚠ CHECK: There was an error processing the request")
    print("     Look at the server output for error details")
    print("     Usually happens with invalid or minimal input")
    
    print("\n4. WARNING: Input is too minimal for proper analysis")
    print("   ✓ NORMAL: The system detected input that was too brief")
    print("     You'll receive guidance on how to improve your input")
    
    print("\n" + "="*60)

def main():
    """Main function to start the server"""
    print("\nStarting Personality Analysis Server with Log Explanation")
    
    # Print the log guide
    print_log_guide()
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser).start()
    
    print("\nStarting server...")
    
    # Start the FastAPI server
    uvicorn.run(
        "app.api:app", 
        host="127.0.0.1", 
        port=8000, 
        log_level="info"
    )

if __name__ == "__main__":
    main()
