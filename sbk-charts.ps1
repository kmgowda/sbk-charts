# Copyright (c) KMG. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

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

$Policy = Read-ProjectPolicy $PolicyFile
$AppName = Get-RequiredPolicyValue $Policy "application.name"
$DistributionName = Get-RequiredPolicyValue $Policy "application.distribution_name"
$AppModule = Get-RequiredPolicyValue $Policy "application.module"
$MinimumPython = Get-RequiredPolicyValue $Policy "runtime.minimum_python"
$ManagedPython = Get-RequiredPolicyValue $Policy "runtime.managed_python"
$ManagedRuntimeName = Get-RequiredPolicyValue $Policy "runtime.managed_runtime_directory"
$LockDirectoryName = Get-RequiredPolicyValue $Policy "runtime.lock_directory"
$BootstrapManager = Get-RequiredPolicyValue $Policy "bootstrap.manager"
$BootstrapManagerVersion = Get-RequiredPolicyValue $Policy "bootstrap.manager_version"
$BootstrapDownloadBaseUrl = Get-RequiredPolicyValue $Policy "bootstrap.download_base_url"
$DefaultCondaEnvironment = Get-RequiredPolicyValue $Policy "runtime.default_conda_environment"
$RuntimeStateName = Get-RequiredPolicyValue $Policy "runtime.runtime_state_file"
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
$SelectedProfile = "core"
$SelectedRequirements = ""
foreach ($ApplicationArgument in $ApplicationArguments) {
    $RequirementPath = [string] $Policy["ai.requirements.$ApplicationArgument"]
    if ($RequirementPath) {
        $SelectedBackend = $ApplicationArgument
        $SelectedProfile = $ApplicationArgument
        $SelectedRequirements = Join-Path $ProjectRoot $RequirementPath
        break
    }
}
$LockFile = Join-Path (Join-Path $ProjectRoot $LockDirectoryName) "$SelectedProfile.txt"
$ManagedArchitecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
$ManagedTarget = if ($ManagedArchitecture -eq [System.Runtime.InteropServices.Architecture]::X64) {
    "windows-amd64"
} elseif ($ManagedArchitecture -eq [System.Runtime.InteropServices.Architecture]::Arm64) {
    "windows-arm64"
} else { "" }
$ExpectedFingerprint = ""
if ($ManagedTarget -and (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
    $FingerprintText = "$ManagedPython`n$ManagedTarget`n$SelectedProfile`n" + `
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
        [string[]] $Arguments
    )
    & $PythonPath $PolicyReader --remember-environment $EnvironmentKind `
        $EnvironmentPrefix $RuntimeStateFile $EnvironmentFingerprint $EnvironmentProfile
    if ($LASTEXITCODE -ne 0) {
        Write-LauncherMessage "WARNING: Could not remember successful $EnvironmentKind environment $EnvironmentPrefix"
    }
    & $PythonPath $PolicyReader --runtime-details $EnvironmentKind $EnvironmentPrefix
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
        [string] $EnvironmentProfile = "core",
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
        Write-LauncherMessage "Reusing managed environment $EnvironmentPrefix"
    } else {
        if (-not (Install-Project $PythonPath "$EnvironmentKind environment $EnvironmentPrefix")) {
            return
        }
    }
    Start-Application -PythonPath $PythonPath -EnvironmentKind $EnvironmentKind `
        -EnvironmentPrefix $EnvironmentPrefix -EnvironmentFingerprint $EnvironmentFingerprint `
        -EnvironmentProfile $EnvironmentProfile -Arguments $Arguments
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

    $TemporaryDirectory = Join-Path $ManagedRuntimeRoot ".tool-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Force -Path $TemporaryDirectory | Out-Null
    $ArchivePath = Join-Path $TemporaryDirectory $ArchiveName
    $ArchiveUrl = "$BootstrapDownloadBaseUrl/$BootstrapManagerVersion/$ArchiveName"
    Write-LauncherMessage "Downloading pinned $BootstrapManager $BootstrapManagerVersion for $ManagedTarget"
    for ($DownloadAttempt = 1; $DownloadAttempt -le 3; $DownloadAttempt++) {
        try {
            Invoke-WebRequest -UseBasicParsing -Uri $ArchiveUrl -OutFile $ArchivePath
            break
        } catch {
            if ($DownloadAttempt -eq 3) { throw }
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
    Remove-Item -LiteralPath $TemporaryDirectory -Recurse -Force
    return $ToolPath
}

function Start-ManagedEnvironment {
    param([string] $EnvironmentPrefix, [string[]] $Arguments)
    if (-not $ManagedTarget) {
        throw "This Windows architecture has no managed runtime. Use a portable release."
    }
    if (-not $ExpectedFingerprint) { throw "Cannot calculate the managed environment fingerprint" }
    if (-not (Test-Path -LiteralPath $LockFile -PathType Leaf)) {
        throw "Dependency lock not found: $LockFile"
    }

    New-Item -ItemType Directory -Force -Path $ManagedRuntimeRoot | Out-Null
    $LockPath = Join-Path $ManagedRuntimeRoot "bootstrap.lock"
    $LockAcquired = $false
    for ($Attempt = 0; $Attempt -lt 60 -and -not $LockAcquired; $Attempt++) {
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
    try {
        if (Test-Path -LiteralPath (Join-Path $EnvironmentPrefix "Scripts\python.exe")) {
            Use-EnvironmentPrefix $EnvironmentPrefix "managed" $Arguments `
                -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
                -EnvironmentProfile $SelectedProfile -Managed
        }
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
        & $UvPath venv --managed-python --seed --python $ManagedPython $TemporaryEnvironment
        if ($LASTEXITCODE -ne 0) { throw "Could not create managed environment" }
        $ManagedPythonPath = Join-Path $TemporaryEnvironment "Scripts\python.exe"
        & $UvPath pip install --python $ManagedPythonPath --require-hashes --requirement $LockFile
        if ($LASTEXITCODE -ne 0) { throw "Could not install locked $SelectedProfile dependencies" }
        & $UvPath pip install --python $ManagedPythonPath --no-build-isolation `
            --no-deps --editable $ProjectRoot
        if ($LASTEXITCODE -ne 0 -or -not (Test-EnvironmentReady $ManagedPythonPath)) {
            throw "Managed environment self-check failed"
        }
        if (Test-Path -LiteralPath $EnvironmentPrefix) {
            $StaleEnvironment = "$EnvironmentPrefix.stale-$PID"
            Move-Item -LiteralPath $EnvironmentPrefix -Destination $StaleEnvironment
            Write-LauncherMessage "Preserved incomplete managed environment as $StaleEnvironment"
        }
        New-Item -ItemType Directory -Force -Path (Split-Path $EnvironmentPrefix) | Out-Null
        Move-Item -LiteralPath $TemporaryEnvironment -Destination $EnvironmentPrefix
    } finally {
        if (Test-Path -LiteralPath $LockPath) {
            Remove-Item -LiteralPath $LockPath -Recurse -Force
        }
    }
    Use-EnvironmentPrefix $EnvironmentPrefix "managed" $Arguments `
        -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
        -EnvironmentProfile $SelectedProfile -Managed
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
        Use-EnvironmentPrefix $PreferredPrefix "managed" $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
            -EnvironmentProfile $SelectedProfile -Managed
    } elseif ($PreferredPrefix -and $PreferredKind -eq "venv") {
        Write-LauncherMessage "Trying remembered venv environment $PreferredPrefix"
        Use-EnvironmentPrefix $PreferredPrefix "venv" $ApplicationArguments `
            -PythonRelativePath "Scripts\python.exe"
    } elseif ($PreferredPrefix -and $PreferredKind -eq "conda") {
        Write-LauncherMessage "Trying remembered Conda environment $PreferredPrefix"
        Use-EnvironmentPrefix $PreferredPrefix "conda" $ApplicationArguments
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
    Use-EnvironmentPrefix $EnvironmentPrefix "venv" $ApplicationArguments `
        -PythonRelativePath "Scripts\python.exe"
}

if (-not $env:SBK_CHARTS_VENV) {
    Use-EnvironmentPrefix $env:CONDA_PREFIX "conda" $ApplicationArguments

    $ProjectEnvironmentCandidates = @($VirtualEnvironmentNames | ForEach-Object {
        Join-Path $ProjectRoot $_
    })
    foreach ($EnvironmentPrefix in $ProjectEnvironmentCandidates) {
        Use-EnvironmentPrefix $EnvironmentPrefix "venv" $ApplicationArguments `
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
        Use-EnvironmentPrefix $CondaPrefix.Trim() "conda" $ApplicationArguments
    }
}

$ManagedEnvironment = Join-Path (Join-Path $ManagedRuntimeRoot "envs") $ExpectedFingerprint
if ($ExpectedFingerprint) {
    Use-EnvironmentPrefix $ManagedEnvironment "managed" $ApplicationArguments `
        -PythonRelativePath "Scripts\python.exe" -EnvironmentFingerprint $ExpectedFingerprint `
        -EnvironmentProfile $SelectedProfile -Managed
}

if (-not $env:SBK_CHARTS_VENV) {
    Start-ManagedEnvironment $ManagedEnvironment $ApplicationArguments
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
            Start-Application -PythonPath $VenvPython -EnvironmentKind "venv" `
                -EnvironmentPrefix $VenvPath -Arguments $ApplicationArguments
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
    Use-EnvironmentPrefix $CondaPrefix.Trim() "conda" $ApplicationArguments
}

if (-not $SupportedLauncher) {
    throw "Python >= $MinimumPython and Conda were not found. Install either one and retry."
}

throw "Python >= $MinimumPython was found, but neither venv nor Conda could create a working environment. Ensure the Python venv module and network access to the package index are available."
