"""
Script to start the server with all question repetition fixes applied.
This ensures the server is running with the latest fixes.
"""

import os
import sys
import subprocess
import importlib.util

def is_module_installed(module_name):
    """Check if a module is installed."""
    return importlib.util.find_spec(module_name) is not None

def ensure_uvicorn_installed():
    """Ensure uvicorn is installed."""
    if not is_module_installed("uvicorn"):
        print("Uvicorn is not installed. Installing it now...")
        subprocess.run([sys.executable, "-m", "pip", "install", "uvicorn"], check=True)
        print("Uvicorn installed successfully.")

def apply_fixes_and_start_server():
    """Apply all question repetition fixes and start the server."""
    # First, make sure we have the right Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.environ["PYTHONPATH"] = current_dir
    
    # Import and apply the fixes directly
    try:
        print("Applying question repetition fixes...")
        from app.apply_question_repetition_fix import apply_fix
        from app.emergency_question_fix import apply_emergency_fix
        
        # Apply the fixes
        apply_fix()
        apply_emergency_fix()
        
        print("✅ All fixes applied successfully!")
        print("Starting server with fixes applied...")
        
        # Start uvicorn server with reload enabled
        # Use sys.executable to ensure we're using the correct Python interpreter
        uvicorn_cmd = ["uvicorn", "app.api:app", "--reload"]
        subprocess.run(uvicorn_cmd)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        print("Please make sure you're running this script from the project root directory.")
        return 1
        
    return 0

if __name__ == "__main__":
    # Ensure uvicorn is installed
    ensure_uvicorn_installed()
    
    # Apply fixes and start server
    sys.exit(apply_fixes_and_start_server())
