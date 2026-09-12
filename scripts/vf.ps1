<#
.SYNOPSIS
    VisionForge developer commands (Windows).
.DESCRIPTION
    GNU make is not installed on this machine, so this script is the primary
    developer entrypoint. A Makefile with the same targets exists for Linux/CI.
.EXAMPLE
    .\scripts\vf.ps1 infra-up
    .\scripts\vf.ps1 check
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'infra-up', 'infra-down', 'infra-reset', 'infra-status',
                 'migrate', 'api', 'web', 'worker-cpu', 'worker-gpu', 'worker-render',
                 'e2e', 'e2e-analysis', 'e2e-edit', 'e2e-llm',
                 'test', 'test-integration', 'lint', 'format', 'typecheck',
                 'contracts', 'web-lint', 'web-build', 'compose-check', 'check', 'help')]
    [string]$Command = 'help'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$ApiDir = Join-Path $Root 'apps\api'
$WebDir = Join-Path $Root 'apps\web'
$Compose = Join-Path $Root 'docker-compose.yml'

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "$Name failed (exit $LASTEXITCODE)" }
}

function Invoke-Api([scriptblock]$Body) {
    Push-Location $ApiDir
    $env:PYTHONPATH = 'src'
    try { & $Body } finally { Pop-Location }
}

function Invoke-Web([scriptblock]$Body) {
    Push-Location $WebDir
    try { & $Body } finally { Pop-Location }
}

switch ($Command) {
    'setup' {
        Invoke-Step 'create venv'      { py -3.11 -m venv (Join-Path $Root '.venv') }
        Invoke-Step 'install backend'  { & $Python -m pip install -e "$ApiDir[dev]" }
        Invoke-Step 'install frontend' { Invoke-Web { pnpm install } }
        Write-Host 'Setup complete. Next: .\scripts\vf.ps1 infra-up' -ForegroundColor Green
    }

    'infra-up' {
        Invoke-Step 'start infrastructure' { docker compose -f $Compose up -d }
        Invoke-Step 'apply migrations'     { Invoke-Api { & $Python -m alembic upgrade head } }
    }
    'infra-down'   { docker compose -f $Compose down }
    'infra-reset'  { docker compose -f $Compose down -v }
    'infra-status' { docker compose -f $Compose ps }
    'migrate'      { Invoke-Api { & $Python -m alembic upgrade head } }

    'api' { Invoke-Api { & $Python -m visionforge } }
    'web' { Invoke-Web { pnpm dev } }
    'worker-cpu' {
        # --pool=solo: FFmpeg is the real concurrency limit on a 6-core laptop,
        # and one job at a time keeps memory and disk predictable.
        Invoke-Api {
            & $Python -m celery -A visionforge.workers.cpu worker `
                -n cpu@%h --pool=solo -Q cpu -l info
        }
    }
    'worker-gpu' {
        # --pool=solo is the process-level GPU mutex: one slot, one model, one
        # inference at a time on a 4 GiB card (ADR-0003, ADR-0008).
        Invoke-Api {
            & $Python -m celery -A visionforge.workers.gpu worker `
                -n gpu@%h --pool=solo -Q gpu -l info
        }
    }
    'e2e' {
        # Needs infra-up, the API and a cpu worker already running.
        & $Python (Join-Path $Root 'scripts\e2e_acceptance.py')
    }
    'e2e-analysis' {
        # Needs infra-up, the API, and both cpu and gpu workers running.
        & $Python (Join-Path $Root 'scripts\e2e_analysis.py')
    }
    'e2e-edit' {
        # Needs infra-up, the API, and both cpu and render workers running.
        & $Python (Join-Path $Root 'scripts\e2e_edit.py')
    }
    'e2e-llm' {
        # Same stack as e2e-edit. Run it twice -- once with LLM_ENABLED=false
        # and once with LLM_ENABLED=true -- because the feature has to be
        # correct both with a provider and without one. The script reads
        # /api/planner/capabilities to discover which it is talking to.
        & $Python (Join-Path $Root 'scripts\e2e_llm.py')
    }

    'test'             { Invoke-Api { & $Python -m pytest -m 'not integration and not gpu' } }
    'test-integration' { Invoke-Api { & $Python -m pytest -m integration } }
    'lint' {
        Invoke-Api { & $Python -m ruff check . ; if ($?) { & $Python -m ruff format --check . } }
    }
    'format' {
        Invoke-Api { & $Python -m ruff check --fix . ; & $Python -m ruff format . }
    }
    'typecheck' { Invoke-Api { & $Python -m mypy } }
    'contracts' { Invoke-Api { & (Join-Path $Root '.venv\Scripts\lint-imports.exe') --config .importlinter } }
    'web-lint'  { Invoke-Web { pnpm lint } }
    'web-build' { Invoke-Web { pnpm build } }
    'compose-check' {
        docker compose -f $Compose config --quiet
        if ($?) { Write-Host 'compose config valid' -ForegroundColor Green }
    }

    'check' {
        Invoke-Step 'ruff check'  { Invoke-Api { & $Python -m ruff check . } }
        Invoke-Step 'ruff format' { Invoke-Api { & $Python -m ruff format --check . } }
        Invoke-Step 'mypy'        { Invoke-Api { & $Python -m mypy } }
        Invoke-Step 'contracts'   { Invoke-Api { & (Join-Path $Root '.venv\Scripts\lint-imports.exe') --config .importlinter } }
        Invoke-Step 'pytest'      { Invoke-Api { & $Python -m pytest -m 'not integration and not gpu' } }
        Invoke-Step 'web lint'    { Invoke-Web { pnpm lint } }
        Invoke-Step 'web build'   { Invoke-Web { pnpm build } }
        Invoke-Step 'compose'     { docker compose -f $Compose config --quiet }
        Write-Host 'All checks passed.' -ForegroundColor Green
    }

    default {
        Write-Host "VisionForge developer commands"
        Write-Host ""
        Write-Host "  Setup"
        Write-Host "    setup              Create the venv and install backend + frontend deps"
        Write-Host ""
        Write-Host "  Infrastructure (postgres, redis, minio)"
        Write-Host "    infra-up           Start containers and apply migrations"
        Write-Host "    infra-down         Stop containers, keep data"
        Write-Host "    infra-reset        Stop containers and DELETE data volumes"
        Write-Host "    infra-status       Show container status"
        Write-Host "    migrate            Apply Alembic migrations"
        Write-Host ""
        Write-Host "  Run"
        Write-Host "    api                Start the API (host/port from .env; default 127.0.0.1:8000)"
        Write-Host "    web                Start the web app on http://localhost:3000"
        Write-Host "    worker-cpu         Start a Celery worker on the cpu queue"
        Write-Host "    worker-gpu         Start a Celery worker on the gpu queue (solo pool)"
        Write-Host "    worker-render      Start a Celery worker on the render queue (solo pool)"
        Write-Host "    e2e                Phase 2 acceptance test (needs api + cpu worker)"
        Write-Host "    e2e-analysis       Phase 3 acceptance test (needs api + both workers)"
        Write-Host "    e2e-edit           Phase 4 acceptance test (needs api + cpu and render workers)"
        Write-Host "    e2e-llm            Phase 5 acceptance test (run with LLM on and off)"
        Write-Host ""
        Write-Host "  Quality"
        Write-Host "    check              Run everything CI runs"
        Write-Host "    test               Unit tests (no infrastructure needed)"
        Write-Host "    test-integration   Integration tests (requires infra-up)"
        Write-Host "    lint / format / typecheck / contracts"
        Write-Host "    web-lint / web-build / compose-check"
    }
}
