@echo off
echo.
echo  MedRAG - Clinical Intelligence Suite
echo  =====================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b
)

echo Installing dependencies...
pip install -r requirements.txt -q

echo.
echo Starting MedRAG backend...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop
echo.

if not exist "data\uploads" mkdir data\uploads
if not exist "data\chroma" mkdir data\chroma

cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
