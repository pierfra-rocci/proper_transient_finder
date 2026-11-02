# Author: Pier-Francesco Rocci
# Developed with assistance from ChatGPT, Claude, and Mistral.

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DistributionPath,
    [string]$PythonPath,
    [string]$EnvironmentPath,
    [string]$Wheelhouse
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
if (-not (Test-Path -LiteralPath $DistributionPath -PathType Leaf)) {
    throw "Distribution not found: $DistributionPath. Run scripts/build.ps1 first."
}
$distribution = (Resolve-Path -LiteralPath $DistributionPath).ProviderPath
if (-not ($distribution.EndsWith('.whl') -or $distribution.EndsWith('.tar.gz'))) {
    throw "Choose a wheel (.whl) or source distribution (.tar.gz)."
}
$indexArgs = @()
if (-not [string]::IsNullOrWhiteSpace($Wheelhouse)) {
    if (-not (Test-Path -LiteralPath $Wheelhouse -PathType Container)) {
        throw "Wheelhouse directory not found: $Wheelhouse"
    }
    $indexArgs = @("--no-index", "--find-links", (Resolve-Path -LiteralPath $Wheelhouse).ProviderPath)
}
$environment = if ([string]::IsNullOrWhiteSpace($EnvironmentPath)) {
    Join-Path ([System.IO.Path]::GetTempPath()) ("proper-transient-finder-install-" + [guid]::NewGuid().ToString('N'))
} else {
    $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($EnvironmentPath)
}
if (Test-Path -LiteralPath $environment) {
    throw "Verification requires a new environment directory; refusing to reuse or overwrite $environment"
}
& $python -m venv $environment
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
$isolatedPython = Join-Path $environment "Scripts\python.exe"
Write-Output "Isolated environment: $environment"
Push-Location ([System.IO.Path]::GetTempPath())
try {
    & $isolatedPython -I -m pip install --no-cache-dir @indexArgs $distribution
    if ($LASTEXITCODE -ne 0) {
        throw "Distribution installation failed. See USER_GUIDE.md for legacy dependency preparation."
    }
    & $isolatedPython -I -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Installed dependency requirements are inconsistent."
    }
    $probe = @'
import json
import sys
from importlib.metadata import distribution
from pathlib import Path
import proper_transient_finder as package

location = Path(package.__file__).resolve()
assert Path(sys.prefix).resolve() in location.parents, location
assert sys.prefix != sys.base_prefix
installed = distribution('proper-transient-finder')
assert installed.version == package.__version__
assert not json.loads(installed.read_text('direct_url.json') or '{}').get('dir_info', {}).get('editable', False)
assert (location.parent / 'py.typed').is_file()
assert package.ProperTransientFinder
assert package.PipelineConfig and package.PipelineResult and package.compare_images
print('Installed API:', package.__version__, location)
'@
    & $isolatedPython -I -c $probe
    if ($LASTEXITCODE -ne 0) {
        throw "Installed API import, metadata, typing, or isolation check failed."
    }
    & (Join-Path $environment "Scripts\proper-transient-finder.exe") --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Installed console entry point failed."
    }
    & $isolatedPython -I -m proper_transient_finder --help | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Installed module entry point failed."
    }
    Write-Output "Runtime-only installation, API, and CLI verified before installing development tools."
    & $isolatedPython -I -m pip install --no-cache-dir @indexArgs "${distribution}[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw "Installation of optional development tools failed."
    }
    & $isolatedPython -I -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Development dependency requirements are inconsistent."
    }
    & $isolatedPython -I -m pytest -c (Join-Path $projectRoot "pyproject.toml") (Join-Path $projectRoot "tests") -q -p no:cacheprovider --basetemp (Join-Path $environment "regression-tests")
    if ($LASTEXITCODE -ne 0) {
        throw "Installed-distribution regression tests failed."
    }
    Write-Output "Installation verified: non-editable package, dependency consistency, API, CLI, and regression suite."
} finally {
    Pop-Location
}