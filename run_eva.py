#!/usr/bin/env python3

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_python_version():
    if sys.version_info < (3, 8):
        print("Python 3.8 or higher is required")
        sys.exit(1)
    print(f" Python {sys.version_info.major}.{sys.version_info.minor} detected")

def check_env_file():
    env_file = Path("gemini.env")
    if not env_file.exists():
        print("gemini.env file not found!")
        print("Please create a gemini.env file with your API keys:")
        print("OPENAI_API_KEY=your_key_here")
        print("GEMINI_API_KEY=your_key_here")
        return False
    with open(env_file, 'r') as f:
        content = f.read()
        if "your_openai_api_key_here" in content or "your_google_gemini_api_key_here" in content:
            print("Please update your API keys in gemini.env")
            print("Replace 'your_*_api_key_here' with actual API keys")
            return False
    
    print("Environment file configured")
    return True

def install_dependencies():
    print("Installing dependencies...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True, capture_output=True)
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False

def check_ports():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(('127.0.0.1', 8000)) == 0:
            print("Port 8000 is already in use")
            response = input("Do you want to kill the existing process? (y/N): ")
            if response.lower() == 'y':
                try:
                    if os.name == 'nt':  # Windows
                        subprocess.run(["taskkill", "/F", "/IM", "python.exe"], capture_output=True)
                    else:  # Unix-like
                        subprocess.run(["pkill", "-f", "uvicorn"], capture_output=True)
                    time.sleep(2)
                    print("Previous process terminated")
                except:
                    print("Failed to terminate existing process")
                    return False
            else:
                return False
        else:
            print("Port 8000 is available")
    return True

def launch_server():
    """Launch the FastAPI server."""
    print("Starting EVA Therapy Bot server...")
    print("Opening browser automatically...")
    
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://127.0.0.1:8000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", "server:app", 
            "--host", "127.0.0.1", "--port", "8000", "--reload"
        ])
    except KeyboardInterrupt:
        print("\n EVA Therapy Bot server stopped")
    except Exception as e:
        print(f"Server error: {e}")

def main():
    checks = [
        ("Python version", check_python_version),
        ("Environment file", check_env_file),
        ("Port availability", check_ports),
    ]
    
    for check_name, check_func in checks:
        print(f"\n🔍 Checking {check_name}...")
        try:
            if callable(check_func):
                result = check_func()
                if result is False:
                    print(f"{check_name} check failed")
                    return
        except Exception as e:
            print(f"Error during {check_name} check: {e}")
            return
    
    # Install dependencies
    print(f"\n Checking dependencies...")
    if not install_dependencies():
        return
    
    # Launch server
    print(f"\n All checks passed! Launching EVA...")
    print(f" Access EVA at: http://127.0.0.1:8000")
    print(f" Press Ctrl+C to stop the server")
    print(f" Make sure to allow microphone and camera permissions")
    
    launch_server()

if __name__ == "__main__":
    main()
