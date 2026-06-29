# start_helios.ps1 — launch the Helios solar-flare API + forecast page.
# Usage:  right-click > Run with PowerShell,  or:  ./start_helios.ps1
# Uses the project's isolated .venv (Python 3.12). Ctrl+C to stop.

$ErrorActionPreference = "Stop"
$proj   = $PSScriptRoot
$venvPy = Join-Path $proj ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPy)) {
    Write-Error "Virtual env not found at $venvPy. Re-create it with: py -3.12 -m venv .venv; .\.venv\Scripts\python -m pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting Helios at http://127.0.0.1:8000/  (Ctrl+C to stop)..." -ForegroundColor Cyan
Start-Process "http://127.0.0.1:8000/"   # open the forecast page in the browser
& $venvPy -m uvicorn api.server:app --host 127.0.0.1 --port 8000
