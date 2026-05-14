@echo off
set BACKEND_DIR=%~dp0
set FRONTEND_DIR=%~dp0..\ai-assistant-web

echo ============================================
echo   AI Assistant - Starting...
echo   Backend : http://localhost:8000/docs
echo   Frontend: http://localhost:5173
echo ============================================

start "AI_Backend" /d "%BACKEND_DIR%" cmd /c uvicorn src.api.main:app --reload --port 8000

if exist "%FRONTEND_DIR%" (
    start "AI_Frontend" /d "%FRONTEND_DIR%" cmd /c npm run dev
) else (
    echo [WARNING] Frontend not found: %FRONTEND_DIR%
)

pause
