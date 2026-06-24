#!/bin/bash
# Run with: bash run-mac-linux.command   (or make it executable: chmod +x it)
# Creates a local Python environment the first time, installs the viewer's
# requirements, then opens it in your browser.
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating Python environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo
echo "Starting the viewer. Leave this window open; close it to stop."
streamlit run app.py
