# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

[CmdletBinding()]
param([string]$PythonPath)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = if ([string]::IsNullOrWhiteSpace($PythonPath)) {
    Join-Path $projectRoot ".venv\Scripts\python.exe"
} else {
    $PythonPath
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python executable not found: $python. Create the project environment or pass -PythonPath explicitly."
}
$python = (Resolve-Path -LiteralPath $python).ProviderPath
& $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The selected interpreter must be Python 3.11 or newer."
}
$targets = @(
    (Join-Path $projectRoot "src")
    (Join-Path $projectRoot "tests")
    (Join-Path $projectRoot "experiments")
    (Join-Path $projectRoot "scripts")
    (Join-Path $projectRoot "run_proper_transient_finder.py")
)

& $python -m ruff check @targets
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python -m ruff format --check @targets
exit $LASTEXITCODE