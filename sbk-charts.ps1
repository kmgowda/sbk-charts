# Copyright (c) KMG. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
##

Set-StrictMode -Version 2.0
$ErrorActionPreference = "Stop"
$ApplicationArguments = @($args)

$ProjectRoot = $PSScriptRoot
$PolicyFile = Join-Path $ProjectRoot "sbk-charts.ini"
$PolicyReader = Join-Path $ProjectRoot "scripts\project_policy.py"

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

function Get-PositivePolicyInteger {
    param(
        [hashtable] $Policy,
        [string] $Key,
        [string] $Description
    )
    $ParsedValue = 0
    $Value = Get-RequiredPolicyValue $Policy $Key
    if (-not [int]::TryParse($Value, [ref] $ParsedValue) -or $ParsedValue -lt 1) {
        throw "$Description must be at least one"
    }
    return $ParsedValue
}

$Policy = Read-ProjectPolicy $PolicyFile
$AppName = Get-RequiredPolicyValue $Policy "application.name"
$DistributionName = Get-RequiredPolicyValue $Policy "application.distribution_name"
$AppModule = Get-RequiredPolicyValue $Policy "application.module"
$VersionFileName = Get-RequiredPolicyValue $Policy "application.version_file"
$MinimumPython = Get-RequiredPolicyValue $Policy "runtime.minimum_python"
$ManagedPython = Get-RequiredPolicyValue $Policy "runtime.managed_python"
$ManagedRuntimeName = Get-RequiredPolicyValue $Policy "runtime.managed_runtime_directory"
$LockDirectoryName = Get-RequiredPolicyValue $Policy "runtime.lock_directory"
$BootstrapLockTimeoutSeconds = Get-PositivePolicyInteger $Policy `
    "runtime.bootstrap_lock_timeout_seconds" "Bootstrap lock timeout in seconds"
$BootstrapManager = Get-RequiredPolicyValue $Policy "bootstrap.manager"
$BootstrapManagerVersion = Get-RequiredPolicyValue $Policy "bootstrap.manager_version"
$BootstrapDownloadBaseUrl = Get-RequiredPolicyValue $Policy "bootstrap.download_base_url"
$BootstrapDownloadTimeoutSeconds = Get-PositivePolicyInteger $Policy `
    "bootstrap.download_timeout_seconds" "Bootstrap download timeout in seconds"
$BootstrapDownloadRetries = Get-PositivePolicyInteger $Policy `
    "bootstrap.download_retries" "Bootstrap download retries"
$DefaultCondaEnvironment = Get-RequiredPolicyValue $Policy "runtime.default_conda_environment"
$DefaultProfile = Get-RequiredPolicyValue $Policy "runtime.default_profile"
$RuntimeStateName = Get-RequiredPolicyValue $Policy "runtime.runtime_state_file"
$RuntimeStateSchema = Get-RequiredPolicyValue $Policy "runtime.runtime_state_schema"
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
$RuntimeStateFile = if ($env:SBK_CHARTS_STATE_FILE) {
    $env:SBK_CHARTS_STATE_FILE
} else {
    Join-Path $ProjectRoot $RuntimeStateName
}
$ManagedRuntimeRoot = if ($env:SBK_CHARTS_RUNTIME_ROOT) {
    $env:SBK_CHARTS_RUNTIME_ROOT
} else {
    Join-Path $ProjectRoot $ManagedRuntimeName
}
$SelectedBackend = ""
$SelectedProfile = $DefaultProfile
$SelectedRequirements = ""
$SkipApplicationValue = $false
foreach ($ApplicationArgument in $ApplicationArguments) {
    if ($SkipApplicationValue) {
        $SkipApplicationValue = $false
        continue
    }
    if ($ApplicationArgument -in @("-i", "--ifiles", "-o", "--ofile", "-secs", "--seconds")) {
        $SkipApplicationValue = $true
        continue
    }
    if ($ApplicationArgument.StartsWith("-")) { continue }
    $RequirementPath = [string] $Policy["ai.requirements.$ApplicationArgument"]
    if ($RequirementPath) {
        $SelectedBackend = $ApplicationArgument
        $SelectedProfile = $ApplicationArgument
        $SelectedRequirements = Join-Path $ProjectRoot $RequirementPath
        break
    }
}
$LockFile = Join-Path (Join-Path $ProjectRoot $LockDirectoryName) "$SelectedProfile.txt"
$ManagedArchitectureKey = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString().ToLowerInvariant()
$ManagedSystem = [string] $Policy["portable.platforms.win32"]
$ManagedArchitecture = [string] $Policy["portable.architectures.$ManagedArchitectureKey"]
$ManagedTarget = if ($ManagedSystem -and $ManagedArchitecture) {
    "$ManagedSystem-$ManagedArchitecture"
} else { "" }
$ExpectedFingerprint = ""
if ($ManagedTarget -and (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
    $FingerprintText = "$ManagedPython`n$ManagedTarget`n$SelectedProfile`n" + `
        (Get-Content -LiteralPath (Join-Path $ProjectRoot $VersionFileName) -Raw) + "`n" + `
        (Get-Content -LiteralPath $LockFile -Raw)
    $Hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        $HashBytes = $Hasher.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($FingerprintText))
        $ExpectedFingerprint = ([BitConverter]::ToString($HashBytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $Hasher.Dispose()
    }
}

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

function Test-PythonLauncherVenv {
    param(
        [string] $CommandPath,
        [string[]] $Prefix
    )
    $ProbeRoot = Join-Path ([System.IO.Path]::GetTempPath()) `
        "sbk-charts-venv-probe-$([guid]::NewGuid().ToString('N'))"
    $ProbeEnvironment = Join-Path $ProbeRoot "venv"
    try {
        New-Item -ItemType Directory -Force -Path $ProbeRoot | Out-Null
        $CreateArguments = @($Prefix) + @("-m", "venv", $ProbeEnvironment)
        & $CommandPath @CreateArguments *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        $ProbePython = Join-Path $ProbeEnvironment "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $ProbePython -PathType Leaf)) { return $false }
        & $ProbePython -m ensurepip --version *> $null
        if ($LASTEXITCODE -ne 0) { return $false }
        & $ProbePython -m pip --version *> $null
        return $LASTEXITCODE -eq 0
    } finally {
        if (Test-Path -LiteralPath $ProbeRoot) {
            Remove-Item -LiteralPath $ProbeRoot -Recurse -Force
        }
    }
}

function Test-EnvironmentReady {
    param([string] $PythonPath)
    if ($SelectedBackend) {
        & $PythonPath $PolicyReader --environment-ready $SelectedBackend 2>$null
    } else {
        & $PythonPath $PolicyReader --environment-ready 2>$null
    }
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
    if ($SelectedRequirements) {
        & $PythonPath -m pip install --requirement $SelectedRequirements
        if ($LASTEXITCODE -ne 0) {
            return $false
        }
    }
    return Test-EnvironmentReady $PythonPath
}

function Start-Application {
    param(
        [string] $PythonPath,
        [string] $EnvironmentKind,
        [string] $EnvironmentPrefix,
        [string] $EnvironmentFingerprint,
        [string] $EnvironmentProfile,
        [string] $SelectionSource,
        [bool] $SavedEnvironmentReused,
        [bool] $EnvironmentCreated,
        [string[]] $Arguments
    )
    if ($EnvironmentFingerprint) {
        & $PythonPath $PolicyReader --remember-environment $EnvironmentKind `
            $EnvironmentPrefix $RuntimeStateFile $EnvironmentFingerprint $EnvironmentProfile
    }
    else {
        # Windows PowerShell 5.1 drops empty native-command arguments. Use the
        # four-value form so a legacy environment can still remember its profile.
        & $PythonPath $PolicyReader --remember-environment $EnvironmentKind `
            $EnvironmentPrefix $RuntimeStateFile $EnvironmentProfile
    }
    if ($LASTEXITCODE -ne 0) {
        Write-LauncherMessage "WARNING: Could not remember successful $EnvironmentKind environment $EnvironmentPrefix"
    }
    $SavedValue = if ($SavedEnvironmentReused) { "yes" } else { "no" }
    $CreatedValue = if ($EnvironmentCreated) { "yes" } else { "no" }
    & $PythonPath $PolicyReader --runtime-details $EnvironmentKind $EnvironmentPrefix `
        $EnvironmentProfile $SelectionSource $SavedValue $CreatedValue
    if ($LASTEXITCODE -ne 0) {
        throw "Could not report runtime details"
    }
    & $PythonPath -m $AppModule @Arguments
    exit $LASTEXITCODE
}

function Use-EnvironmentPrefix {
    param(
        [string] $EnvironmentPrefix,
        [string] $EnvironmentKind,
        [string[]] $Arguments,
        [string] $PythonRelativePath = "python.exe",
        [string] $EnvironmentFingerprint = "",
        [string] $EnvironmentProfile = $DefaultProfile,
        [string] $SelectionSource = "unknown",
        [switch] $SavedEnvironmentReused,
        [switch] $EnvironmentCreated,
        [switch] $Managed
    )
    if (-not $EnvironmentPrefix) {
        return
    }
    $PythonPath = Join-Path $EnvironmentPrefix $PythonRelativePath
    if (-not (Test-SupportedPython $PythonPath)) {
        return
    }
    if ($Managed) {
        if (-not (Test-EnvironmentReady $PythonPath)) { return }
        if ($EnvironmentCreated) {
            Write-LauncherMessage "Using newly created managed environment $EnvironmentPrefix"
        } else {
            Write-LauncherMessage "Reusing managed environment $EnvironmentPrefix"
        }
    } else {
        if (-not (Install-Project $PythonPath "$EnvironmentKind environment $EnvironmentPrefix")) {
            return
        }
    }
    Start-Application -PythonPath $PythonPath -EnvironmentKind $EnvironmentKind `
        -EnvironmentPrefix $EnvironmentPrefix -EnvironmentFingerprint $EnvironmentFingerprint `
        -EnvironmentProfile $EnvironmentProfile -SelectionSource $SelectionSource `
        -SavedEnvironmentReused $SavedEnvironmentReused.IsPresent `
        -EnvironmentCreated $EnvironmentCreated.IsPresent -Arguments $Arguments
}

function Get-BootstrapManager {
    $ArchiveName = Get-RequiredPolicyValue $Policy "bootstrap.$ManagedTarget-archive"
    $ArchiveSha = Get-RequiredPolicyValue $Policy "bootstrap.$ManagedTarget-sha256"
    $ToolDirectory = Join-Path (Join-Path $ManagedRuntimeRoot "tools") `
        "$BootstrapManager-$BootstrapManagerVersion"
    $ToolPath = Join-Path $ToolDirectory "$BootstrapManager.exe"
    $ChecksumMarker = Join-Path $ToolDirectory "archive.sha256"
    $BinaryChecksumMarker = Join-Path $ToolDirectory "binary.sha256"
    if ((Test-Path -LiteralPath $ToolPath -PathType Leaf) -and
        (Test-Path -LiteralPath $ChecksumMarker -PathType Leaf) -and
        (Test-Path -LiteralPath $BinaryChecksumMarker -PathType Leaf) -and
        ((Get-Content -LiteralPath $ChecksumMarker -Raw).Trim() -eq $ArchiveSha)) {
        $ExpectedBinarySha = (Get-Content -LiteralPath $BinaryChecksumMarker -Raw).Trim()
        $ActualBinarySha = (Get-FileHash -LiteralPath $ToolPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualBinarySha -eq $ExpectedBinarySha) { return $ToolPath }
    }

    $TemporaryDirectory = $null
    try {
        $TemporaryDirectory = Join-Path $ManagedRuntimeRoot ".tool-$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Force -Path $TemporaryDirectory | Out-Null
        $ArchivePath = Join-Path $TemporaryDirectory $ArchiveName
        $ArchiveUrl = "$BootstrapDownloadBaseUrl/$BootstrapManagerVersion/$ArchiveName"
        Write-LauncherMessage "Downloading pinned $BootstrapManager $BootstrapManagerVersion for $ManagedTarget"
        for ($DownloadAttempt = 1; $DownloadAttempt -le $BootstrapDownloadRetries; $DownloadAttempt++) {
            try {
                Invoke-WebRequest -UseBasicParsing -Uri $ArchiveUrl -OutFile $ArchivePath `
                    -TimeoutSec $BootstrapDownloadTimeoutSeconds
                break
            } catch {
                if ($DownloadAttempt -eq $BootstrapDownloadRetries) { throw }
                Start-Sleep -Seconds $DownloadAttempt
            }
        }
        $ActualSha = (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualSha -ne $ArchiveSha) {
            throw "Downloaded bootstrap manager failed SHA-256 verification"
        }
        Expand-Archive -LiteralPath $ArchivePath -DestinationPath $TemporaryDirectory -Force
        $ExtractedTool = Get-ChildItem -LiteralPath $TemporaryDirectory -Filter "$BootstrapManager.exe" `
            -File -Recurse | Select-Object -First 1
        if (-not $ExtractedTool) { throw "Bootstrap manager executable is missing from $ArchiveName" }
        New-Item -ItemType Directory -Force -Path $ToolDirectory | Out-Null
        Copy-Item -LiteralPath $ExtractedTool.FullName -Destination $ToolPath -Force
        Set-Content -LiteralPath $ChecksumMarker -Value $ArchiveSha -Encoding ASCII
        $BinarySha = (Get-FileHash -LiteralPath $ToolPath -Algorithm SHA256).Hash.ToLowerInvariant()
        Set-Content -LiteralPath $BinaryChecksumMarker -Value $BinarySha -Encoding ASCII
        return $ToolPath
    } finally {
        if ($TemporaryDirectory -and (Test-Path -LiteralPath $TemporaryDirectory)) {
            Remove-Item -LiteralPath $TemporaryDirectory -Recurse -Force
        }
    }
}

function Start-ManagedEnvironment {
    param([string] $EnvironmentPrefix, [string[]] $Arguments)
    if (-not $ManagedTarget) {
        Write-LauncherMessage "This Windows architecture has no managed runtime. Use a portable release or install Python/Conda."
        return
    }
    if (-not $ExpectedFingerprint) { throw "Cannot calculate the managed environment fingerprint" }
    if (-not (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
        throw "Dependency lock not found: $LockFile"
    }

    New-Item -ItemType Directory -Force -Path $ManagedRuntimeRoot | Out-Null
    $LockPath = Join-Path $ManagedRuntimeRoot "bootstrap.lock"
    $LockAcquired = $false
    for ($Attempt = 0; $Attempt -lt $BootstrapLockTimeoutSeconds -and -not $LockAcquired; $Attempt++) {
        try {
            New-Item -ItemType Directory -Path $LockPath -ErrorAction Stop | Out-Null
            $LockAcquired = $true
        } catch {
            $OwnerFile = Join-Path $LockPath "pid"
            if (Test-Path -LiteralPath $OwnerFile -PathType Leaf) {
                $OwnerPid = (Get-Content -LiteralPath $OwnerFile -Raw).Trim()
                if ($OwnerPid -match '^\d+$' -and
                    -not (Get-Process -Id ([int] $OwnerPid) -ErrorAction SilentlyContinue)) {
                    Remove-Item -LiteralPath $LockPath -Recurse -Force
                    continue
                }
            }
            Start-Sleep -Seconds 1
        }
    }
    if (-not $LockAcquired) { throw "Timed out waiting for bootstrap lock $LockPath" }
    Set-Content -LiteralPath (Join-Path $LockPath "pid") -Value $PID -Encoding ASCII
    $TemporaryEnvironment = $null
    $EnvironmentPublishedByAnotherProcess = $false
    try {
        $PublishedPython = Join-Path $EnvironmentPrefix "Scripts\python.exe"
        if ((Test-SupportedPython $PublishedPython) -and
            (Test-EnvironmentReady $PublishedPython)) {
            $EnvironmentPublishedByAnotherProcess = $true
        } else {
            $UvPath = if ($env:SBK_CHARTS_UV) { $env:SBK_CHARTS_UV } else { Get-BootstrapManager }
            if (-not (Test-Path -LiteralPath $UvPath -PathType Leaf)) {
                throw "Bootstrap manager is not executable: $UvPath"
            }
            $PythonInstallDirectory = Join-Path $ManagedRuntimeRoot "python"
            $env:UV_PYTHON_INSTALL_DIR = $PythonInstallDirectory
            $env:UV_CACHE_DIR = Join-Path $ManagedRuntimeRoot "cache"
            $env:UV_LINK_MODE = "copy"
            Write-LauncherMessage "Installing managed Python $ManagedPython"
            & $UvPath python install --install-dir $PythonInstallDirectory $ManagedPython
            if ($LASTEXITCODE -ne 0) { throw "Could not install managed Python $ManagedPython" }
            $TemporaryEnvironment = Join-Path $ManagedRuntimeRoot `
                ".env-$ExpectedFingerprint-$([guid]::NewGuid().ToString('N'))"
            & $UvPath venv --relocatable --managed-python --seed --python $ManagedPython $TemporaryEnvironment
            if ($LASTEXITCODE -ne 0) { throw "Could not create managed environment" }
            $ManagedPythonPath = Join-Path $TemporaryEnvironment "Scripts\python.exe"
            & $UvPath pip install --python $ManagedPythonPath --require-hashes --requirement $LockFile
            if ($LASTEXITCODE -ne 0) { throw "Could not install locked $SelectedProfile dependencies" }
            & $UvPath pip install --python $ManagedPythonPath --no-build-isolation `
                --no-deps $ProjectRoot
            if ($LASTEXITCODE -ne 0 -or -not (Test-EnvironmentReady $ManagedPythonPath)) {
                throw "Managed environment self-check failed"
            }
            if (Test-Path -LiteralPath $EnvironmentPrefix) {
                Remove-Item -LiteralPath $EnvironmentPrefix -Recurse -Force
            }
            New-Item -ItemType Directory -Force -Path (Split-Path $EnvironmentPrefix) | Out-Null
            Move-Item -LiteralPath $TemporaryEnvironment -Destination $EnvironmentPrefix
            $TemporaryEnvironment = $null
        }
    } finally {
        if ($TemporaryEnvironment -and (Test-Path -LiteralPath $TemporaryEnvironment)) {
            Remove-Item -LiteralPath $TemporaryEnvironment -Recurse -Force
        }
        if (Test-Path -LiteralPath $LockPath) {
            Remove-Item -LiteralPath $LockPath -Recurse -Force
        }
    }
    if ($EnvironmentPublishedByAnotherProcess) {
        Use-EnvironmentPrefix -EnvironmentPrefix $EnvironmentPrefix -EnvironmentKind "managed" `
            -Arguments $Arguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
            -EnvironmentProfile $SelectedProfile -SelectionSource "managed-cache" -Managed
    } else {
        Use-EnvironmentPrefix -EnvironmentPrefix $EnvironmentPrefix -EnvironmentKind "managed" `
            -Arguments $Arguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
            -EnvironmentProfile $SelectedProfile -SelectionSource "created-managed" `
            -EnvironmentCreated -Managed
    }
    throw "Managed environment could not start $AppName"
}

function Read-RuntimeState {
    if (-not (Test-Path -LiteralPath $RuntimeStateFile -PathType Leaf)) {
        return @{}
    }
    $Values = @{}
    foreach ($RawLine in Get-Content -LiteralPath $RuntimeStateFile) {
        if ($RawLine -match '^([^=]+)=(.*)$') {
            $Values[$Matches[1]] = $Matches[2]
        }
    }
    if ($Values.ContainsKey("schema") -and $Values["schema"] -ne $RuntimeStateSchema) {
        Write-LauncherMessage "Ignoring runtime state with unsupported schema $($Values['schema'])"
        return @{}
    }
    return $Values
}

if (-not $env:SBK_CHARTS_VENV) {
    $RuntimeState = Read-RuntimeState
    $PreferredKind = [string] $RuntimeState["kind"]
    $PreferredPrefix = [string] $RuntimeState["prefix"]
    $PreferredFingerprint = [string] $RuntimeState["fingerprint"]
    $PreferredProfile = [string] $RuntimeState["profile"]
    if ($PreferredPrefix -and $PreferredKind -eq "managed" -and
        $PreferredFingerprint -eq $ExpectedFingerprint -and
        $PreferredProfile -eq $SelectedProfile) {
        Write-LauncherMessage "Trying remembered managed environment $PreferredPrefix"
        Use-EnvironmentPrefix -EnvironmentPrefix $PreferredPrefix -EnvironmentKind "managed" `
            -Arguments $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
            -EnvironmentProfile $SelectedProfile -SelectionSource "saved-state" `
            -SavedEnvironmentReused -Managed
    } elseif ($PreferredPrefix -and $PreferredKind -eq "venv") {
        Write-LauncherMessage "Trying remembered venv environment $PreferredPrefix"
        Use-EnvironmentPrefix -EnvironmentPrefix $PreferredPrefix -EnvironmentKind "venv" `
            -Arguments $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentProfile $SelectedProfile `
            -SelectionSource "saved-state" -SavedEnvironmentReused
    } elseif ($PreferredPrefix -and $PreferredKind -eq "conda") {
        Write-LauncherMessage "Trying remembered Conda environment $PreferredPrefix"
        Use-EnvironmentPrefix -EnvironmentPrefix $PreferredPrefix -EnvironmentKind "conda" `
            -Arguments $ApplicationArguments `
            -EnvironmentProfile $SelectedProfile -SelectionSource "saved-state" `
            -SavedEnvironmentReused
    }
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
    $CandidateSource = if ($env:SBK_CHARTS_VENV) { "explicit-venv" } else { "active-venv" }
    Use-EnvironmentPrefix -EnvironmentPrefix $EnvironmentPrefix -EnvironmentKind "venv" `
        -Arguments $ApplicationArguments `
        -PythonRelativePath "Scripts\python.exe" -EnvironmentProfile $SelectedProfile `
        -SelectionSource $CandidateSource
}

if (-not $env:SBK_CHARTS_VENV) {
    Use-EnvironmentPrefix -EnvironmentPrefix $env:CONDA_PREFIX -EnvironmentKind "conda" `
        -Arguments $ApplicationArguments `
        -EnvironmentProfile $SelectedProfile -SelectionSource "active-conda"

    $ProjectEnvironmentCandidates = @($VirtualEnvironmentNames | ForEach-Object {
        Join-Path $ProjectRoot $_
    })
    foreach ($EnvironmentPrefix in $ProjectEnvironmentCandidates) {
        Use-EnvironmentPrefix -EnvironmentPrefix $EnvironmentPrefix -EnvironmentKind "venv" `
            -Arguments $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentProfile $SelectedProfile `
            -SelectionSource "project-venv"
    }
}

$CondaCommand = Get-Command conda -ErrorAction SilentlyContinue
$NamedCondaEnvironmentExists = $false
if ($CondaCommand) {
    $CondaPrefixOutput = & $CondaCommand.Source run --name $CondaEnvironmentName python -c "import sys; print(sys.prefix)" 2>$null
    if ($LASTEXITCODE -eq 0 -and $CondaPrefixOutput) {
        $NamedCondaEnvironmentExists = $true
        $CondaPrefix = [string](@($CondaPrefixOutput) | Select-Object -Last 1)
    }
}

$ManagedEnvironment = Join-Path (Join-Path $ManagedRuntimeRoot "envs") $ExpectedFingerprint
if ($ExpectedFingerprint) {
    Use-EnvironmentPrefix -EnvironmentPrefix $ManagedEnvironment -EnvironmentKind "managed" `
        -Arguments $ApplicationArguments `
        -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
        -EnvironmentProfile $SelectedProfile -SelectionSource "managed-cache" -Managed
}

if ($NamedCondaEnvironmentExists) {
    Use-EnvironmentPrefix -EnvironmentPrefix ($CondaPrefix.Trim()) -EnvironmentKind "conda" `
        -Arguments $ApplicationArguments `
        -EnvironmentProfile $SelectedProfile -SelectionSource "named-conda"
}

if (-not $env:SBK_CHARTS_VENV -and $ExpectedFingerprint) {
    try {
        Start-ManagedEnvironment -EnvironmentPrefix $ManagedEnvironment -Arguments $ApplicationArguments
    } catch {
        Write-LauncherMessage "Managed environment setup failed: $($_.Exception.Message)"
        Write-LauncherMessage "Trying legacy Python/Conda"
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
        if (Test-PythonLauncherVenv -CommandPath $Command.Source -Prefix @($Launcher.Prefix)) {
            $SupportedLauncher = @{ Command = $Command.Source; Prefix = @($Launcher.Prefix) }
            break
        }
        Write-LauncherMessage "Python candidate $($Command.Source) cannot create a working venv; trying the next candidate"
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
            Start-Application -PythonPath $VenvPython -EnvironmentKind "venv" `
                -EnvironmentPrefix $VenvPath -EnvironmentFingerprint "" `
                -EnvironmentProfile $SelectedProfile -SelectionSource "created-venv" `
                -SavedEnvironmentReused $false -EnvironmentCreated $true `
                -Arguments $ApplicationArguments
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
    if ($NamedCondaEnvironmentExists) {
        Use-EnvironmentPrefix -EnvironmentPrefix ($CondaPrefix.Trim()) -EnvironmentKind "conda" `
            -Arguments $ApplicationArguments `
            -EnvironmentProfile $SelectedProfile -SelectionSource "named-conda"
    } else {
        Use-EnvironmentPrefix -EnvironmentPrefix ($CondaPrefix.Trim()) -EnvironmentKind "conda" `
            -Arguments $ApplicationArguments `
            -EnvironmentProfile $SelectedProfile -SelectionSource "created-conda" `
            -EnvironmentCreated
    }
}

if (-not $SupportedLauncher) {
    throw "Python >= $MinimumPython and Conda were not found. Install either one and retry."
}

throw "Python >= $MinimumPython was found, but neither venv nor Conda could create a working environment. Ensure the Python venv module and network access to the package index are available."
