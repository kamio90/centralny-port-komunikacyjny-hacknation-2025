#!/bin/bash

echo "🏗️ Setting up CPK Classifier for Apple M4 Max..."
echo "=================================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "📦 Conda not found. Installing Miniconda..."
    echo "Downloading Miniconda for ARM64..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    bash Miniconda3-latest-MacOSX-arm64.sh -b
    source ~/miniconda3/bin/activate
    echo "✅ Miniconda installed"
else
    echo "✅ Conda found"
fi

# Create conda environment
echo ""
echo "📦 Creating conda environment 'cpk'..."
conda create -n cpk python=3.10 -y

# Activate environment
echo ""
echo "🔧 Activating environment..."
eval "$(conda shell.bash hook)"
conda activate cpk

# Install PyTorch with MPS support
echo ""
echo "🤖 Installing PyTorch with MPS support..."
pip install torch torchvision torchaudio

# Install other dependencies
echo ""
echo "📚 Installing other dependencies..."
pip install -r requirements.txt

# Test MPS availability
echo ""
echo "🧪 Testing MPS availability..."
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ MPS available: {torch.backends.mps.is_available()}')"

# Create symlink to reference data
echo ""
echo "🔗 Setting up data directory..."
mkdir -p data/input
if [ -f "hackaton_task_files/Chmura_Zadanie_Dane.las" ]; then
    echo "✅ Reference LAS file found"
else
    echo "⚠️  Reference LAS file not found at hackaton_task_files/Chmura_Zadanie_Dane.las"
fi

echo ""
echo "=================================================="
echo "✅ Setup complete!"
echo ""
echo "To start the application:"
echo "  conda activate cpk"
echo "  streamlit run app.py"
echo ""
echo "The app will open at http://localhost:8501"
echo "=================================================="
