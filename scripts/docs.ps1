# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

[CmdletBinding(DefaultParameterSetName = "Generate")]
param(
    [string]$PythonPath,
    [switch]$Check,
    [Parameter(ParameterSetName = "Build", Mandatory = $true)]
    [switch]$Build,
    [Parameter(ParameterSetName = "Serve", Mandatory = $true)]
    [switch]$Serve,
    [Parameter(ParameterSetName = "Serve")]
    [ValidatePattern('^127\.0\.0\.1:\d{1,5}$')]
    [string]$Address = "127.0.0.1:8001"
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
& $python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 13) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "The selected interpreter must be Python 3.13 or newer."
}
$arguments = @("-I", "-S", (Join-Path $PSScriptRoot "generate_docs.py"))
if ($Check) {
    $arguments += "--check"
}
& $python @arguments
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if ($Build) {
    & $python -m mkdocs build --strict --config-file (Join-Path $projectRoot "mkdocs.yml")
} elseif ($Serve) {
    & $python -m mkdocs serve --config-file (Join-Path $projectRoot "mkdocs.yml") --dev-addr $Address
}
exit $LASTEXITCODE