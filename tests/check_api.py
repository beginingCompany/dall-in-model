"""
Diagnostic utility for checking API server status and configuration
"""
import os
import sys
import requests
import json
from pprint import pprint

def check_server_status(base_url="http://127.0.0.1:8000"):
    """Check if the server is running and responding"""
    print("\n=== Checking Server Status ===")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is running (uptime: {data.get('uptime', 0):.1f}s)")
            print(f"   Version: {data.get('version', 'unknown')}")
            return True
        else:
            print(f"⚠️ Server returned status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        return False
    except Exception as e:
        print(f"❌ Error checking server status: {str(e)}")
        return False

def test_minimal_input(base_url="http://127.0.0.1:8000"):
    """Test how the server handles minimal input"""
    print("\n=== Testing Minimal Input Handling ===")
    
    minimal_input = {
        "id": 999,
        "user_input": "I take short breaks to clear my mind.",
        "new_input": [],
        "languages": ["en"]
    }
    
    try:
        print("Sending minimal input...")
        response = requests.post(
            f"{base_url}/analyze-personality",
            json=minimal_input,
            timeout=10
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.text:
            try:
                result = response.json()
                print(f"Response status: {result.get('status', 'unknown')}")
                print(f"Contains fallback questions: {'clarification_questions' in result}")
                print(f"Missing traits identified: {result.get('missing_traits', [])}")
                return True
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response (length: {len(response.text)})")
                print(f"Response text: {response.text[:100]}...")
                return False
        else:
            print("❌ Empty response received")
            return False
    except Exception as e:
        print(f"❌ Error testing minimal input: {str(e)}")
        return False

def test_good_input(base_url="http://127.0.0.1:8000"):
    """Test how the server handles good input"""
    print("\n=== Testing Good Input Handling ===")
    
    good_input = {
        "id": 888,
        "user_input": """I'm a software engineer who enjoys solving complex problems. 
        I'm generally calm and patient, but can get frustrated with inefficient processes. 
        In social settings, I'm somewhat reserved at first but become more outgoing once I'm comfortable. 
        I enjoy outdoor activities and reading in my free time.""",
        "new_input": [
            {
                "question": "How do you handle stress?",
                "answer": "I try to break problems into smaller parts and tackle them one by one."
            }
        ],
        "languages": ["en"]
    }
    
    try:
        print("Sending good input...")
        response = requests.post(
            f"{base_url}/analyze-personality",
            json=good_input,
            timeout=20
        )
        
        print(f"Status code: {response.status_code}")
        
        if response.text:
            try:
                result = response.json()
                print(f"Response status: {result.get('status', 'unknown')}")
                print(f"Description length: {len(result.get('description_english', ''))}")
                print(f"Missing traits: {result.get('missing_traits', [])}")
                return True
            except json.JSONDecodeError:
                print(f"❌ Invalid JSON response (length: {len(response.text)})")
                print(f"Response text: {response.text[:100]}...")
                return False
        else:
            print("❌ Empty response received")
            return False
    except Exception as e:
        print(f"❌ Error testing good input: {str(e)}")
        return False

def check_static_files():
    """Check if static files are properly set up"""
    print("\n=== Checking Static Files ===")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(project_root, "static")
    
    required_files = [
        os.path.join(static_dir, "index.html"),
        os.path.join(static_dir, "debug.js")
    ]
    
    all_ok = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found {os.path.basename(file_path)}")
        else:
            print(f"❌ Missing {os.path.basename(file_path)}")
            all_ok = False
    
    return all_ok

def main():
    """Run diagnostic checks"""
    print("=== API Server Diagnostic Utility ===")
    
    base_url = "http://127.0.0.1:8000"  # Default URL
    
    # Check if server is running
    server_ok = check_server_status(base_url)
    
    if not server_ok:
        print("\n⚠️ Server is not running. Start it with 'python run_with_ui.py' before continuing.")
        print("\nDo you want to check static files anyway? (y/n)")
        response = input("> ").strip().lower()
        if response != 'y':
            return
    
    # Check static files
    static_ok = check_static_files()
    
    if server_ok:
        # Test minimal input handling
        minimal_ok = test_minimal_input(base_url)
        
        # Test good input handling
        good_ok = test_good_input(base_url)
        
        # Overall assessment
        print("\n=== Diagnostic Summary ===")
        print(f"Server Status: {'✅ OK' if server_ok else '❌ Not Running'}")
        print(f"Static Files: {'✅ OK' if static_ok else '❌ Issues Found'}")
        print(f"Minimal Input Handling: {'✅ OK' if minimal_ok else '❌ Issues Found'}")
        print(f"Good Input Handling: {'✅ OK' if good_ok else '❌ Issues Found'}")
        
        if not minimal_ok or not good_ok:
            print("\n⚠️ Issues found with API handling. Try running the server with:")
            print("python debug_server.py")
            print("\nAnd check the console output for errors.")

if __name__ == "__main__":
    main()
