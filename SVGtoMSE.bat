@echo off
REM Call the Python script using the current directory
python "%~dp0SVGtoMSE.py" %*
REM Pause so you can read any output before the window closes
pause