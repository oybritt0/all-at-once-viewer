@echo off
REM Double-click to launch. Creates a local Python environment the first time,
REM installs what the viewer needs, then opens it in your browser.
cd /d "%~dp0"
where py >nul 2>nul && (set PYEXE=py) || (set PYEXE=python)
if not exist .venv (
  echo Creating Python environment...
  %PYEXE% -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Starting the viewer. Leave this window open; close it to stop.
streamlit run app.py
pause
