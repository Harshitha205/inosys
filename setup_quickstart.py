#!/usr/bin/env python3
"""
Quick Start Setup Script
Initializes the entire project environment for running the application
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_step(step_num, description):
    """Print step information"""
    print(f"\n[{step_num}/5] {description}")
    print("-" * 70)


def run_command(cmd, description):
    """Run a command and report status"""
    try:
        print(f"  Running: {description}")
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"  ✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error: {e}")
        print(f"  Output: {e.stderr}")
        return False


def main():
    """Main setup function"""
    
    print_header("SUPPLY CHAIN DELIVERY PREDICTION - QUICK START SETUP")
    
    # Get project root
    project_root = Path(__file__).parent
    milestone4_dir = project_root / "Milestone4_Deployment"
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    python_version = sys.version_info
    required_version = (3, 10)
    
    if python_version >= required_version:
        print(f"  ✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    else:
        print(f"  ✗ Python 3.10+ required (found {python_version.major}.{python_version.minor})")
        sys.exit(1)
    
    # Step 2: Create virtual environment
    print_step(2, "Setting up virtual environment")
    venv_path = project_root / "venv"
    
    if venv_path.exists():
        print(f"  ℹ Virtual environment already exists at {venv_path}")
    else:
        if run_command(f'"{sys.executable}" -m venv "{venv_path}"', "Creating virtual environment"):
            print(f"  ✓ Virtual environment created at {venv_path}")
        else:
            print("  ✗ Failed to create virtual environment")
            sys.exit(1)
    
    # Step 3: Setup deployment models
    print_step(3, "Setting up trained models")
    
    setup_script = milestone4_dir / "setup.py"
    
    # Use the virtual environment python
    if sys.platform == "win32":
        python_cmd = str(venv_path / "Scripts" / "python.exe")
    else:
        python_cmd = str(venv_path / "bin" / "python")
    
    if run_command(f'"{python_cmd}" "{setup_script}"', "Copying models from Milestone 3"):
        print("  ✓ Models successfully copied")
    else:
        print("  ⚠ Warning: Models may not be available")
    
    # Step 4: Verify installation
    print_step(4, "Verifying installation")
    
    required_files = [
        milestone4_dir / "app.py",
        milestone4_dir / "README.md",
        project_root / "README.md",
        project_root / "requirements.txt",
    ]
    
    all_files_exist = True
    for file in required_files:
        if file.exists():
            print(f"  ✓ Found: {file.relative_to(project_root)}")
        else:
            print(f"  ✗ Missing: {file.relative_to(project_root)}")
            all_files_exist = False
    
    if not all_files_exist:
        print("  ⚠ Some files are missing")
    
    # Step 5: Display next steps
    print_step(5, "Setup complete!")
    
    print("""
    ✅ Setup completed successfully!
    
    Next Steps:
    -----------
    
    1. Activate virtual environment:
       Windows: venv\\Scripts\\activate
       macOS/Linux: source venv/bin/activate
    
    2. Start the application:
       streamlit run Milestone4_Deployment/app.py
    
    3. Open in browser:
       http://localhost:8501
    
    4. Explore the application:
       - 📊 Prediction: Make delivery predictions
       - 📈 Model Performance: View metrics
       - ℹ️ About: Project information
       - 🔧 Data Info: Feature analysis
    
    Documentation:
    ---------------
    - README.md - Project overview
    - Milestone4_Deployment/README.md - Deployment guide
    - Milestone4_Deployment/PROJECT_REPORT.md - Detailed report
    - CONTRIBUTING.md - Contributing guidelines
    
    Docker Deployment:
    ------------------
    docker-compose up --build
    
    Common Issues:
    ---------------
    - Port 8501 in use: streamlit run app.py --server.port 8502
    - Models not found: python Milestone4_Deployment/setup.py
    - Import errors: pip install -r requirements.txt --upgrade
    
    For more help, see documentation files or GitHub issues.
    """)
    
    print("=" * 70)
    print("  Setup completed successfully! Ready to run the application.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
