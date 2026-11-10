@echo off
title UNIVERSAL CRYPTO ARB PLATFORM
echo.
echo  ============================================================
echo   UNIVERSAL CRYPTO ARBITRAGE HFT PLATFORM
echo   Paper Mode: ON (default - no real orders)
echo  ============================================================
echo.

:: ---- Locate Python ----
set PYTHON=python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python not found in PATH. Install Python 3.11+ and add to PATH.
    pause
    exit /b 1
)

set BACKEND_DIR=%~dp0backend
set FRONTEND_DIR=%~dp0frontend
set BACKEND_PORT=8901
set FRONTEND_PORT=5900

:: ---- Kill any old instances ----
echo [CLEAN] Stopping any existing instances...
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%BACKEND_PORT% " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr ":%FRONTEND_PORT% " ^| findstr "LISTENING"') do (
    taskkill /PID %%a /F >nul 2>&1
)

:: ---- Copy .env.example if no .env ----
if not exist "%BACKEND_DIR%\.env" (
    if exist "%~dp0.env.example" (
        copy "%~dp0.env.example" "%BACKEND_DIR%\.env" >nul
        echo [INFO] Created backend\.env from .env.example
    )
)

:: ---- Install backend deps (first run) ----
if not exist "%BACKEND_DIR%\venv" (
    echo [SETUP] Creating Python virtual environment...
    %PYTHON% -m venv "%BACKEND_DIR%\venv"
    echo [SETUP] Installing backend requirements...
    "%BACKEND_DIR%\venv\Scripts\pip" install -r "%BACKEND_DIR%\requirements.txt"
    echo [SETUP] Backend ready.
)

:: ---- Install frontend deps (first run) ----
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [SETUP] Installing frontend dependencies...
    pushd "%FRONTEND_DIR%"
    call npm install
    popd
    echo [SETUP] Frontend ready.
)

:: ---- Start Backend ----
echo [START] Starting backend on http://localhost:%BACKEND_PORT% ...
start "CRYPTO ARB Backend" cmd /k "cd /d %BACKEND_DIR% && venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port %BACKEND_PORT%"

:: ---- Start Frontend ----
echo [START] Starting frontend on http://localhost:%FRONTEND_PORT% ...
start "CRYPTO ARB Frontend" cmd /k "cd /d %FRONTEND_DIR% && npm run dev"

:: ---- Wait and open browser ----
echo.
echo [INFO] Waiting 8 seconds for servers to start...
timeout /t 8 /nobreak >nul
start http://localhost:%FRONTEND_PORT%

echo.
echo  ============================================================
echo   RUNNING:
echo     Backend API:  http://localhost:%BACKEND_PORT%
echo     Frontend UI:  http://localhost:%FRONTEND_PORT%
echo     API Docs:     http://localhost:%BACKEND_PORT%/api/docs
echo  ============================================================
echo.
echo   Paper Mode is ON. Binance+Coinbase via native WebSocket.
echo   Kraken+OKX via CCXT REST polling. NO real orders placed.
echo.
echo   Pages: Dashboard, Exchanges, Markets, Opportunities,
echo          Price Spread, Auto Trade, Execution, Risk
echo.
echo   Press any key to exit this launcher (servers keep running)
pause >nul
