#!/usr/bin/env python3
"""
Research Assistant Application Launcher
Run this script to start the Research Assistant web application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'streamlit', 'langchain', 'langchain-openai', 'langchain-chroma',
        'langchain-tavily', 'chromadb', 'sentence-transformers', 'tavily-python',
        'pymupdf', 'rank-bm25', 'python-dotenv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n💡 Install missing packages with:")
        print("pip install -r requirements.txt")
        return False
    
    return True

def check_config():
    """Check if configuration is set up"""
    config_path = Path("app/config.py")
    
    if not config_path.exists():
        print("❌ Configuration file not found!")
        print("💡 Copy config_example.py to app/config.py and add your API keys")
        return False
    
    # Check if API keys are configured
    try:
        sys.path.insert(0, str(Path("app").absolute()))
        from config import OPENAI_API_KEY, TAVILY_API_KEY
        
        if OPENAI_API_KEY == "your_openai_api_key_here":
            print("⚠️  OpenAI API key not configured in config.py")
        
        if TAVILY_API_KEY == "your_tavily_api_key_here":
            print("⚠️  Tavily API key not configured in config.py")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error importing config: {e}")
        return False

def main():
    """Main function to run the application"""
    print("🔍 Research Assistant - Starting Application")
    print("=" * 50)
    
    # Check requirements
    print("📦 Checking requirements...")
    if not check_requirements():
        sys.exit(1)
    print("✅ All required packages are installed")
    
    # Check configuration
    print("🔧 Checking configuration...")
    if not check_config():
        sys.exit(1)
    print("✅ Configuration loaded")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("✅ Data directory ready")
    
    # Start Streamlit app
    print("🚀 Starting Research Assistant...")
    print("🌐 Opening web browser...")
    
    # Run streamlit app
    app_path = Path("app/main.py")
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.address", "localhost",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "light"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Research Assistant stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 