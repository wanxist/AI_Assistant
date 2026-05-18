@echo off
set BACKEND_DIR=%~dp0
set FRONTEND_DIR=%~dp0..\ai-assistant-web

echo ============================================
echo   AI Assistant - Starting...
echo ============================================

:: Clear stale Python bytecode cache (prevents old code from running)
echo   Clearing __pycache__...
for /d /r "%BACKEND_DIR%" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul

:: Prevent Python from creating new bytecode cache files
set PYTHONDONTWRITEBYTECODE=1

:: Start backend
echo   Backend : http://localhost:8000/docs
start "AI_Backend" /d "%BACKEND_DIR%" cmd /k "set PYTHONDONTWRITEBYTECODE=1 && uvicorn src.api.main:app --reload --port 8000"

:: Start frontend
if exist "%FRONTEND_DIR%" (
    echo   Frontend: http://localhost:5173
    start "AI_Frontend" /d "%FRONTEND_DIR%" cmd /k npm run dev
) else (
    echo   [WARNING] Frontend not found: %FRONTEND_DIR%
)

echo ============================================
pause
