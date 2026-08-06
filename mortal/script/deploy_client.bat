@echo off
setlocal
chcp 65001 >nul

REM source dir: mortal/ (this bat lives in mortal/script/)
set "SRC=%~dp0..\"
set "OUT=%~dp0out\mortal"

echo Cleaning %OUT% ...
if exist "%OUT%" rmdir /s /q "%OUT%"

echo Creating layout ...
mkdir "%OUT%\conf"
mkdir "%OUT%\baseline_v1"

echo Copying python source ...
copy "%SRC%client.py" "%OUT%" >nul || goto :fail
copy "%SRC%player.py" "%OUT%" >nul || goto :fail
copy "%SRC%engine.py" "%OUT%" >nul || goto :fail
copy "%SRC%model.py" "%OUT%" >nul || goto :fail
copy "%SRC%common.py" "%OUT%" >nul || goto :fail
copy "%SRC%config.py" "%OUT%" >nul || goto :fail
copy "%SRC%config_v2.py" "%OUT%" >nul || goto :fail
copy "%SRC%prelude.py" "%OUT%" >nul || goto :fail
copy "%SRC%config.toml" "%OUT%" >nul || goto :fail
copy "%SRC%setup_client.bat" "%OUT%" >nul || goto :fail
copy "%SRC%_setup_client.py" "%OUT%" >nul || goto :fail

echo Copying compiled extension ...
copy "%SRC%libriichi.pyd" "%OUT%" >nul || goto :fail

echo Copying config files ...
copy "%SRC%conf\*.toml" "%OUT%\conf" >nul || goto :fail

echo Copying baseline checkpoint ...
copy "%SRC%baseline_v1\mortal.pth" "%OUT%\baseline_v1" >nul || goto :fail

echo.
echo Done. Deploy package is at: %OUT%
echo On target machine: run setup_client.bat first, then python client.py
exit /b 0

:fail
echo.
echo ERROR: copy failed, package is incomplete.
exit /b 1
