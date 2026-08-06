@echo off
setlocal enabledelayedexpansion

REM ============================================================
REM Client environment setup script
REM Patches config_v2.py paths to current directory, disables AMP
REM Usage: run setup_client.bat, then python client.py
REM ============================================================

REM Get deploy root directory (where this bat lives)
set "ROOT=%~dp0"
REM Remove trailing backslash
set "ROOT=!ROOT:~0,-1!"

echo Deploy root: %ROOT%
echo.

REM Ask for server IP
set "SERVER_IP=192.168.1.239"
set /p "SERVER_IP=Enter server IP [%SERVER_IP%]: "
if "!SERVER_IP!"=="" set "SERVER_IP=192.168.1.239"

REM Ask for GPU device
set "DEVICE=cuda:0"
set /p "DEVICE=Enter GPU device [%DEVICE%]: "
if "!DEVICE!"=="" set "DEVICE=cuda:0"

echo.
echo Patching config files ...
python "%ROOT%\_setup_client.py" "%ROOT%" "%SERVER_IP%" "%DEVICE%"

if errorlevel 1 (
    echo.
    echo ERROR: config patch failed. Make sure Python is in PATH.
    exit /b 1
)

echo.
echo Done. Run client:  python client.py
exit /b 0
