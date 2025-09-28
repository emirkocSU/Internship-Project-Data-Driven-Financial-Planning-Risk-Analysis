@echo off
REM Setup script for Elips Financial Planner (Windows)

echo 🏢 Elips Financial Planner - Setup Script
echo ==========================================

REM Check Python version
echo 📋 Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.8 or higher
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo ✅ Python %python_version% detected

REM Create virtual environment
echo 🔧 Creating virtual environment...
if exist venv (
    echo ⚠️  Virtual environment already exists. Removing...
    rmdir /s /q venv
)

python -m venv venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)
echo ✅ Virtual environment created

REM Activate virtual environment
echo ⚡ Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install dependencies
echo 📚 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Install development dependencies if available
if exist requirements-dev.txt (
    echo 🛠️  Installing development dependencies...
    pip install -r requirements-dev.txt
)

REM Install package in development mode
echo 🔧 Installing package in development mode...
pip install -e .
if errorlevel 1 (
    echo ⚠️  Package installation failed, but continuing...
)

REM Run initial test
echo 🧪 Running initial system test...
python main.py > setup_test.log 2>&1

if errorlevel 1 (
    echo ❌ Initial test failed. Check setup_test.log for details.
    pause
    exit /b 1
) else (
    echo ✅ Initial test passed!
    echo 📝 Test output saved to setup_test.log
)

echo.
echo 🎉 Setup completed successfully!
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate
echo.
echo To run the application:
echo   python main.py
echo.
echo To run tests:
echo   pytest
echo.
echo To deactivate the environment:
echo   deactivate

pause