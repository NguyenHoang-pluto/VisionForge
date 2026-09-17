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
                 'e2e', 'e2e-analysis', 'e2e-edit', 'e2e-llm', 'e2e-editor', 'e2e-music',
                 'e2e-style', 'e2e-effects', 'e2e-coedit', 'e2e-editorial',
                 'test', 'test-integration', 'lint', 'format', 'typecheck',
                 'contracts', 'web-lint', 'web-build', 'compose-check', 'check', 'help')]
    [string]$Command = 'help',

    # Anything after the command, handed to the script it runs. Only the
    # acceptance scripts take arguments, and only to narrow what they do --
    # `e2e-editorial football` runs one scenario instead of three, which is how
    # a machine with 7.4 GB gets through them.
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
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
    'worker-render' {
        # Rendering is one long FFmpeg process per job, so --pool=solo gets that
        # process the cores it can use and keeps two encodes off the same
        # scratch disk. Phase 4 renders do not start without this worker.
        Invoke-Api {
            & $Python -m celery -A visionforge.workers.render worker `
                -n render@%h --pool=solo -Q render -l info
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

    'e2e-editor' {
        # Phase 6. Same stack as e2e-edit: plans an edit, edits the timeline the
        # way the editor does, stores it through the manual route, renders that
        # and verifies the file against what the timeline said.
        & $Python (Join-Path $Root 'scripts\e2e_editor.py')
    }

    'e2e-music' {
        # Phase 7. Same stack again: detects the tempo of a locally generated
        # track, plans a beat-synced cut under it, mixes the audio and verifies
        # the MP4's audio stream independently with ffprobe.
        & $Python (Join-Path $Root 'scripts\e2e_music.py')
    }

    'e2e-style' {
        # Phase 8. Measures a reference video, dials its influence in, and
        # plans an edit under it.
        & $Python (Join-Path $Root 'scripts\e2e_style.py')
    }

    'e2e-effects' {
        # Phase 9. Transitions, effects and burned-in subtitles, proved visible
        # by comparing frames against the same edit rendered without them.
        & $Python (Join-Path $Root 'scripts\e2e_effects.py')
    }

    'e2e-coedit' {
        # Phase 10. An edit changed by asking: delta, validation, a new version,
        # undo that restores byte for byte, and a render of the patched plan.
        # The deterministic path needs no AI provider -- run it a second time
        # with LLM_ENABLED=true to exercise the model path as well.
        & $Python (Join-Path $Root 'scripts\e2e_coedit.py')
    }

    'e2e-editorial' {
        # Phase 11. The creative auto-editing engine: twelve clips analysed,
        # given editorial events and story roles, selected for quality and
        # diversity, paced on an energy curve, cut under a genre policy and
        # rendered. Three scenarios -- football, nature, gaming -- and on a
        # small machine they are meant to be run one at a time:
        #     .\scriptsf.ps1 e2e-editorial football
        & $Python (Join-Path $Root 'scripts\e2e_editorial.py') @Rest
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
        Write-Host "    e2e-editor         Phase 6 acceptance test (timeline editing -> render)"
        Write-Host "    e2e-music          Phase 7 acceptance test (beats, audio mix -> render)"
        Write-Host "    e2e-style          Phase 8 acceptance test (reference style -> plan)"
        Write-Host "    e2e-effects        Phase 9 acceptance test (transitions, effects, subtitles)"
        Write-Host "    e2e-coedit         Phase 10 acceptance test (AI co-editor, versions, undo)"
        Write-Host "    e2e-editorial      Phase 11 acceptance test (editorial engine -> render)"
        Write-Host ""
        Write-Host "  Quality"
        Write-Host "    check              Run everything CI runs"
        Write-Host "    test               Unit tests (no infrastructure needed)"
        Write-Host "    test-integration   Integration tests (requires infra-up)"
        Write-Host "    lint / format / typecheck / contracts"
        Write-Host "    web-lint / web-build / compose-check"
    }
}
