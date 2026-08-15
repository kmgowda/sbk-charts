# Copyright (c) KMG. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ApplicationArguments = @($args)

$ProjectRoot = $PSScriptRoot
$PolicyFile = Join-Path $ProjectRoot "sbk-charts.ini"

function Read-ProjectPolicy {
    param([string] $Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Runtime policy not found: $Path"
    }
    $Values = @{}
    $Section = ""
    foreach ($RawLine in Get-Content -LiteralPath $Path) {
        $Line = $RawLine.Trim()
        if (-not $Line -or $Line.StartsWith("#") -or $Line.StartsWith(";")) {
            continue
        }
        if ($Line -match '^\[(.+)\]$') {
            $Section = $Matches[1]
            continue
        }
        if ($Line -match '^([^=]+)=(.*)$') {
            $Key = $Matches[1].Trim()
            $Values["$Section.$Key"] = $Matches[2].Trim()
        }
    }
    return $Values
}

function Get-RequiredPolicyValue {
    param(
        [hashtable] $Policy,
        [string] $Key
    )
    $Value = [string] $Policy[$Key]
    if (-not $Value) {
        throw "Runtime policy value is missing: $Key"
    }
    return $Value
}

$Policy = Read-ProjectPolicy $PolicyFile
$AppName = Get-RequiredPolicyValue $Policy "application.name"
$DistributionName = Get-RequiredPolicyValue $Policy "application.distribution_name"
$AppModule = Get-RequiredPolicyValue $Policy "application.module"
$MinimumPython = Get-RequiredPolicyValue $Policy "runtime.minimum_python"
$DefaultCondaEnvironment = Get-RequiredPolicyValue $Policy "runtime.default_conda_environment"
$VirtualEnvironmentNames = @(
    (Get-RequiredPolicyValue $Policy "runtime.virtual_environment_names").Split(',') |
        ForEach-Object { $_.Trim() } | Where-Object { $_ }
)
$WindowsPythonLaunchers = @(
    (Get-RequiredPolicyValue $Policy "runtime.windows_python_launchers").Split(',') |
        ForEach-Object { $_.Trim() } | Where-Object { $_ }
)
$CondaEnvironmentName = if ($env:SBK_CHARTS_CONDA_ENV) {
    $env:SBK_CHARTS_CONDA_ENV
} else {
    $DefaultCondaEnvironment
}
$DefaultVenv = Join-Path $ProjectRoot $VirtualEnvironmentNames[0]

function Write-LauncherMessage {
    param([string] $Message)
    [Console]::Error.WriteLine("${AppName}: $Message")
}

function Test-SupportedPython {
    param([string] $PythonPath)
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        return $false
    }
    & $PythonPath -c "import sys; required=tuple(map(int, sys.argv[1].split('.'))); raise SystemExit(0 if sys.version_info >= required else 1)" $MinimumPython 2>$null
    return $LASTEXITCODE -eq 0
}

function Test-EnvironmentReady {
    param([string] $PythonPath)
    & $PythonPath -c "import importlib, sys; from importlib.metadata import version; version(sys.argv[1]); importlib.import_module(sys.argv[2])" $DistributionName $AppModule 2>$null
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    & $PythonPath -m pip check *> $null
    return $LASTEXITCODE -eq 0
}

function Install-Project {
    param(
        [string] $PythonPath,
        [string] $EnvironmentLabel
    )
    if (Test-EnvironmentReady $PythonPath) {
        Write-LauncherMessage "Reusing $EnvironmentLabel"
        return $true
    }

    Write-LauncherMessage "Installing $AppName into $EnvironmentLabel"
    & $PythonPath -m pip install --editable $ProjectRoot
    if ($LASTEXITCODE -ne 0) {
        return $false
    }
    return Test-EnvironmentReady $PythonPath
}

function Start-Application {
    param(
        [string] $PythonPath,
        [string[]] $Arguments
    )
    & $PythonPath -m $AppModule @Arguments
    exit $LASTEXITCODE
}

function Use-EnvironmentPrefix {
    param(
        [string] $EnvironmentPrefix,
        [string] $EnvironmentKind,
        [string[]] $Arguments,
        [string] $PythonRelativePath = "python.exe"
    )
    if (-not $EnvironmentPrefix) {
        return
    }
    $PythonPath = Join-Path $EnvironmentPrefix $PythonRelativePath
    if (-not (Test-SupportedPython $PythonPath)) {
        return
    }
    if (-not (Install-Project $PythonPath "$EnvironmentKind $EnvironmentPrefix")) {
        return
    }
    Start-Application $PythonPath $Arguments
}

$EnvironmentCandidates = if ($env:SBK_CHARTS_VENV) {
    @($env:SBK_CHARTS_VENV)
} else {
    @($env:VIRTUAL_ENV)
}

$SeenCandidates = @{}
foreach ($EnvironmentPrefix in $EnvironmentCandidates) {
    if (-not $EnvironmentPrefix) {
        continue
    }
    $CandidateKey = $EnvironmentPrefix.ToLowerInvariant()
    if ($SeenCandidates.ContainsKey($CandidateKey)) {
        continue
    }
    $SeenCandidates[$CandidateKey] = $true
    Use-EnvironmentPrefix $EnvironmentPrefix "virtual environment" $ApplicationArguments `
        -PythonRelativePath "Scripts\python.exe"
}

if (-not $env:SBK_CHARTS_VENV) {
    Use-EnvironmentPrefix $env:CONDA_PREFIX "Conda environment" $ApplicationArguments

    $ProjectEnvironmentCandidates = @($VirtualEnvironmentNames | ForEach-Object {
        Join-Path $ProjectRoot $_
    })
    foreach ($EnvironmentPrefix in $ProjectEnvironmentCandidates) {
        Use-EnvironmentPrefix $EnvironmentPrefix "virtual environment" $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe"
    }
}

$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
$NamedCondaEnvironmentExists = $false
if ($CondaCommand) {
    $CondaPrefixOutput = & $CondaCommand.Source run --name $CondaEnvironmentName python -c "import sys; print(sys.prefix)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $CondaPrefixOutput) {
        $NamedCondaEnvironmentExists = $true
        $CondaPrefix = [string](@($CondaPrefixOutput) | Select-Object -Last 1)
        Use-EnvironmentPrefix $CondaPrefix.Trim() "Conda environment" $ApplicationArguments
    }
}

$VenvPath = if ($env:SBK_CHARTS_VENV) { $env:SBK_CHARTS_VENV } else { $DefaultVenv }
$PythonLaunchers = @($WindowsPythonLaunchers | ForEach-Object {
    $Prefix = if ($_ -eq "py") { @("-$MinimumPython") } else { @() }
    @{ Command = $_; Prefix = $Prefix }
})
$SupportedLauncher = $null

foreach ($Launcher in $PythonLaunchers) {
    $Command = Get-Command $Launcher.Command -ErrorAction SilentlyContinue
    if (-not $Command) {
        continue
    }
    $ProbeArguments = @($Launcher.Prefix) + @(
        "-c",
        "import sys; required=tuple(map(int, sys.argv[1].split('.'))); raise SystemExit(0 if sys.version_info >= required else 1)",
        $MinimumPython
    )
    & $Command.Source @ProbeArguments 2>$null
    if ($LASTEXITCODE -eq 0) {
        $SupportedLauncher = @{ Command = $Command.Source; Prefix = @($Launcher.Prefix) }
        break
    }
}

if ($SupportedLauncher) {
    Write-LauncherMessage "Creating virtual environment $VenvPath"
    $CreateArguments = @($SupportedLauncher.Prefix) + @("-m", "venv", $VenvPath)
    & $SupportedLauncher.Command @CreateArguments
    if ($LASTEXITCODE -eq 0) {
        $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
        if ((Test-SupportedPython $VenvPython) -and
            (Install-Project $VenvPython "virtual environment $VenvPath")) {
            Start-Application $VenvPython $ApplicationArguments
        }
    }
    Write-LauncherMessage "Virtual environment setup failed; trying Conda"
}

if ($CondaCommand) {
    if ($NamedCondaEnvironmentExists) {
        Write-LauncherMessage "Updating Conda environment $CondaEnvironmentName with Python >= $MinimumPython"
        & $CondaCommand.Source install --yes --name $CondaEnvironmentName "python>=$MinimumPython" pip
    } else {
        Write-LauncherMessage "Creating Conda environment $CondaEnvironmentName with Python >= $MinimumPython"
        & $CondaCommand.Source create --yes --name $CondaEnvironmentName "python>=$MinimumPython" pip
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Conda could not prepare environment $CondaEnvironmentName"
    }
    $CondaPrefixOutput = & $CondaCommand.Source run --name $CondaEnvironmentName python -c "import sys; print(sys.prefix)"
    $CondaPrefix = [string](@($CondaPrefixOutput) | Select-Object -Last 1)
    Use-EnvironmentPrefix $CondaPrefix.Trim() "Conda environment" $ApplicationArguments
}

if (-not $SupportedLauncher) {
    throw "Python >= $MinimumPython and Conda were not found. Install either one and retry."
}

throw "Python >= $MinimumPython was found, but neither venv nor Conda could create a working environment. Ensure the Python venv module and network access to the package index are available."
