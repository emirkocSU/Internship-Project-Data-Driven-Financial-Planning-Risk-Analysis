#!/bin/bash
# Setup script for Elips Financial Planner (Linux/macOS)

set -e  # Exit on any error

echo "🏢 Elips Financial Planner - Setup Script"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

echo "📋 Checking Python version..."
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "✅ Python $python_version detected (>= $required_version required)"
else
    echo "❌ Python $python_version detected, but >= $required_version required"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing..."
    rm -rf venv
fi

python3 -m venv venv
echo "✅ Virtual environment created"

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies if available
if [ -f "requirements-dev.txt" ]; then
    echo "🛠️  Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Install package in development mode
echo "🔧 Installing package in development mode..."
pip install -e .

# Run initial test
echo "🧪 Running initial system test..."
python main.py > setup_test.log 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Initial test passed!"
    echo "📝 Test output saved to setup_test.log"
else
    echo "❌ Initial test failed. Check setup_test.log for details."
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application:"
echo "  python main.py"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"