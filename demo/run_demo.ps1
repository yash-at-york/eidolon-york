# Eidolon - Conference Demo Launcher (PowerShell)
# Run from the project root: .\demo\run_demo.ps1

param(
    [switch]$InfraOnly,
    [switch]$SkipInfra
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Host ""
Write-Host "  Eidolon - Privacy-Preserving Agentic Debugger" -ForegroundColor Cyan
Write-Host "  Conference Demo Mode" -ForegroundColor Cyan
Write-Host ""

# --- Step 1: Check .env -------------------------------------------------------
if (-not (Test-Path ".env")) {
    Write-Host "[!] .env not found - copying from .env.example" -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "    Edit .env and set your HF_TOKEN before running!" -ForegroundColor Red
    exit 1
}

$env:GHOST_DEMO_MODE = "true"

# --- Step 2: Start Docker infrastructure --------------------------------------
if (-not $SkipInfra) {
    Write-Host "[1/4] Starting infrastructure (Qdrant + FalkorDB + NATS)..." -ForegroundColor Green
    docker compose -f infrastructure/docker-compose.yml up -d
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[!] Docker Compose failed. Is Docker Desktop running?" -ForegroundColor Red
        exit 1
    }
    Write-Host "      Waiting 5s for services to be healthy..." -ForegroundColor DarkGray
    Start-Sleep -Seconds 5
}

if ($InfraOnly) {
    Write-Host "Infrastructure started. Exiting (-InfraOnly)." -ForegroundColor Green
    exit 0
}

# --- Step 3: Start sync worker in background ----------------------------------
Write-Host "[2/4] Starting sync worker..." -ForegroundColor Green
$syncJob = Start-Job -ScriptBlock {
    Set-Location $using:ROOT
    & "$using:ROOT\.venv\Scripts\python.exe" -m src.cloud.sync_worker
}

# --- Step 4: Inject the demo bug ----------------------------------------------
Write-Host "[3/4] Injecting demo bug into demo/app_test.py..." -ForegroundColor Green
& .venv\Scripts\python.exe demo\inject_bug.py

# --- Step 5: Start Eidolon daemon in demo mode -----------------------------
Write-Host "[4/4] Starting Eidolon daemon in DEMO MODE..." -ForegroundColor Green
Write-Host ""
Write-Host "  When the daemon fires, you will see the ghost payload below." -ForegroundColor Yellow
Write-Host "  Then in a NEW terminal, run:" -ForegroundColor Yellow
Write-Host "  .venv\Scripts\python.exe src/agent/graph.py --event `"401 Unauthorized on POST /verify-token`" --service demo-svc" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Press Ctrl+C to stop and restore the original file." -ForegroundColor DarkGray
Write-Host ""

try {
    & .venv\Scripts\python.exe src\ghost_daemon.py --demo-mode --watch demo\app_test.py
} finally {
    Write-Host ""
    Write-Host "Restoring demo/app_test.py..." -ForegroundColor Yellow
    & .venv\Scripts\python.exe demo\inject_bug.py --restore
    Stop-Job $syncJob -ErrorAction SilentlyContinue
    Remove-Job $syncJob -ErrorAction SilentlyContinue
    Write-Host "Demo complete. Clean exit." -ForegroundColor Green
}
