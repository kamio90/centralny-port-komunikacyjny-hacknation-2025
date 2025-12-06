#!/bin/bash

echo "🏗️ Setting up CPK Classifier with pip (simpler approach)..."
echo "============================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "📍 Found Python: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with MPS support (for Apple Silicon)
echo ""
echo "🤖 Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install other dependencies
echo ""
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Test installations
echo ""
echo "🧪 Testing installations..."
python3 << EOF
import sys
print(f"✅ Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ MPS available: {torch.backends.mps.is_available()}")
except ImportError:
    print("❌ PyTorch not installed correctly")

try:
    import streamlit
    print(f"✅ Streamlit version: {streamlit.__version__}")
except ImportError:
    print("❌ Streamlit not installed correctly")

try:
    import laspy
    print(f"✅ laspy installed")
except ImportError:
    print("❌ laspy not installed correctly")
EOF

# Check reference file
echo ""
echo "🔍 Checking for reference data file..."
if [ -f "hackaton_task_files/Chmura_Zadanie_Dane.las" ]; then
    FILE_SIZE=$(ls -lh hackaton_task_files/Chmura_Zadanie_Dane.las | awk '{print $5}')
    echo "✅ Reference LAS file found (${FILE_SIZE})"
else
    echo "⚠️  Reference LAS file not found at hackaton_task_files/Chmura_Zadanie_Dane.las"
    echo "    (You can still use the app with uploaded files)"
fi

echo ""
echo "============================================================"
echo "✅ Setup complete!"
echo ""
echo "To activate the environment and run the app:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"
echo ""
echo "The app will open at http://localhost:8501"
echo ""
echo "To deactivate the environment later:"
echo "  deactivate"
echo "============================================================"
