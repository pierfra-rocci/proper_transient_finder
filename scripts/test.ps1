# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

[CmdletBinding()]
param(
    [string]$PythonPath,
    [string[]]$PytestArgs = @()
)

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

& $python -m pytest -c (Join-Path $projectRoot "pyproject.toml") (Join-Path $projectRoot "tests") @PytestArgs
exit $LASTEXITCODE