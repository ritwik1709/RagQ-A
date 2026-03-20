@echo off
REM Start the FastAPI backend server on Windows

echo Starting Document Q&A Application...
echo.
echo Starting FastAPI Backend on http://localhost:8000
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop the server
echo.

.venv\Scripts\python.exe -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
pause
