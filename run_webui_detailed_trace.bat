@echo off
REM ===========================================================
REM Launch the Detailed Tracing variant of the Latent Explorer.
REM This is a separate copy of app.py (app_detailed_trace.py)
REM so it can run alongside the main app without disrupting it.
REM Runs on port 8503 (main app uses 8501, constellation 8502).
REM Place this file in the project root, alongside
REM app_detailed_trace.py (and app.py, notebooks/, data/).
REM ===========================================================

cd /d "%~dp0"

echo Checking dependencies...
python -m pip install --quiet streamlit plotly umap-learn scikit-learn matplotlib pandas pillow numpy hdbscan
if errorlevel 1 (
    echo.
    echo Dependency install failed. Make sure Python is on PATH.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit. The detailed-tracing UI will open in your
echo browser at http://localhost:8503
echo.
echo Stop the server with Ctrl+C in this window.
echo.

python -m streamlit run app_detailed_trace.py --server.port 8503

pause
