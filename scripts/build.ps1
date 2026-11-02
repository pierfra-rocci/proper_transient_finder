# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

[CmdletBinding()]
param(
    [string]$PythonPath,
    [string]$OutputDirectory,
    [switch]$NoIsolation,
    [switch]$IncludeDependencies
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
$destination = if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    Join-Path $projectRoot "dist"
} else {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
}
$buildArgs = @("-m", "build", "--outdir", $destination, $projectRoot)
if ($NoIsolation) {
    $buildArgs += "--no-isolation"
}
& $python @buildArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if ($IncludeDependencies) {
    $wheelhouse = Join-Path $destination "wheelhouse"
    & $python -m pip wheel --no-build-isolation --no-deps --wheel-dir $wheelhouse "tinynpydb==0.1.2" "properimage==0.7.2"
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    & $python -m pip wheel --only-binary=:all: --find-links $wheelhouse --wheel-dir $wheelhouse "${projectRoot}[dev]"
}
exit $LASTEXITCODE