# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [ValidateSet('Install', 'Repair', 'Verify', 'Run', 'Menu')]
    [string]$Action = 'Menu',
    [string]$WorkRoot = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'HAGISILTA-work'),
    [ValidateSet('auto', 'ko', 'en')]
    [string]$Language = 'auto',
    [switch]$WaitForKey
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$EnvName = 'hagisilta-dlc-3.0'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$EnvironmentFile = Join-Path $RepoRoot 'environment.yml'
$RuntimeScript = Join-Path $PSScriptRoot 'hagisilta_runtime.py'
$AppScript = Join-Path $PSScriptRoot 'hagisilta_platform.py'
$LauncherScript = Join-Path $PSScriptRoot 'hagisilta_launcher.py'
$StateRoot = Join-Path $env:LOCALAPPDATA 'HAGISILTA'
$MiniforgeRoot = Join-Path $StateRoot 'miniforge3'
$CondaExe = Join-Path $MiniforgeRoot 'Scripts\conda.exe'
$LogRoot = Join-Path $StateRoot 'logs'


# This installer must also run before Python is available. Its catalogs are
# separate from the Python application's Korean-source translation catalogs.
function Import-InstallerCatalogs {
    $script:InstallerCatalogs = @{}
    foreach ($code in @('en', 'ko')) {
        $catalogFile = Join-Path $RepoRoot "assets\i18n\installer\$code.json"
        $catalog = Get-Content -LiteralPath $catalogFile -Raw -Encoding UTF8 | ConvertFrom-Json
        $script:InstallerCatalogs[$code] = @{}
        foreach ($property in $catalog.PSObject.Properties) {
            $script:InstallerCatalogs[$code][$property.Name] = [string]$property.Value
        }
    }
}

function Get-InstallerText([string]$Key, [object[]]$Values = @()) {
    if (-not $script:InstallerCatalogs[$script:InstallerLanguage].ContainsKey($Key)) {
        throw "Missing installer translation: $Key"
    }
    $text = $script:InstallerCatalogs[$script:InstallerLanguage][$Key]
    if ($Values.Count -gt 0) { return ($text -f $Values) }
    return $text
}

function Resolve-InstallerLanguage([string]$Requested, [string]$SelectedAction, [bool]$AllowPrompt) {
    if ($Requested -in @('ko', 'en')) {
        return [pscustomobject]@{ Language = $Requested.ToLowerInvariant(); SavePreference = $true }
    }
    $inherited = [string]$env:HAGISILTA_UI_LANGUAGE
    $inherited = $inherited.Trim().ToLowerInvariant()
    if ($inherited -in @('ko', 'en')) {
        return [pscustomobject]@{ Language = $inherited; SavePreference = $false }
    }
    $settingsFile = $env:HAGISILTA_SETTINGS_FILE
    if ([string]::IsNullOrWhiteSpace($settingsFile)) { $settingsFile = Join-Path $StateRoot 'machine-settings.json' }
    try {
        if (Test-Path -LiteralPath $settingsFile -PathType Leaf) {
            $saved = Get-Content -LiteralPath $settingsFile -Raw -Encoding UTF8 | ConvertFrom-Json
            $savedLanguage = ([string]$saved.ui_language).Trim().ToLowerInvariant()
            if ($savedLanguage -in @('ko', 'en')) {
                return [pscustomobject]@{ Language = $savedLanguage; SavePreference = $false }
            }
        }
    } catch { }
    # Run is also invoked by a hidden desktop launcher: never prompt there.
    if ($AllowPrompt -and $SelectedAction -ne 'Run') {
        $prompt = $script:InstallerCatalogs.en.choose_language + ' / ' + $script:InstallerCatalogs.ko.choose_language
        Write-Host $prompt
        Write-Host ('1. ' + $script:InstallerCatalogs.ko.language_name)
        Write-Host ('2. ' + $script:InstallerCatalogs.en.language_name)
        while ($true) {
            $answer = Read-Host ($prompt + ' [1/2, Enter=1]')
            if ([string]::IsNullOrWhiteSpace($answer) -or $answer.Trim() -eq '1') {
                return [pscustomobject]@{ Language = 'ko'; SavePreference = $true }
            }
            if ($answer.Trim() -eq '2') {
                return [pscustomobject]@{ Language = 'en'; SavePreference = $true }
            }
            Write-Host ($script:InstallerCatalogs.en.language_invalid + ' / ' + $script:InstallerCatalogs.ko.language_invalid)
        }
    }
    return [pscustomobject]@{ Language = 'ko'; SavePreference = $false }
}

function Initialize-InstallerLanguage {
    Import-InstallerCatalogs
    $allowPrompt = $false
    try { $allowPrompt = [Environment]::UserInteractive -and -not [Console]::IsInputRedirected } catch { }
    $selection = Resolve-InstallerLanguage $Language $Action $allowPrompt
    $script:InstallerLanguage = $selection.Language
    $script:SaveLanguagePreference = $selection.SavePreference
    $env:HAGISILTA_UI_LANGUAGE = $script:InstallerLanguage
    try { $Host.UI.RawUI.WindowTitle = Get-InstallerText ('title_' + $Action) } catch { }
}

function Save-InstallerLanguage {
    if (-not $script:SaveLanguagePreference) { return }
    $python = Get-EnvironmentPython
    if (-not $python) { return }
    # Reuse the application's atomic, read/merge/write settings API and honor
    # HAGISILTA_SETTINGS_FILE for both reads and writes in isolated checks.
    $saveCode = 'import sys; sys.path.insert(0, sys.argv[1]); from hagisilta_i18n import save_language; save_language(sys.argv[2])'
    & $python -X utf8 -c $saveCode $PSScriptRoot $script:InstallerLanguage
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'language_save_failed') }
    $script:SaveLanguagePreference = $false
}

function Write-Step([string]$Text) {
    Write-Host "`n[$Text]" -ForegroundColor Cyan
}

function Find-Conda {
    if (Test-Path -LiteralPath $CondaExe) { return $CondaExe }
    return $null
}

function Get-EnvironmentPython {
    $conda = Find-Conda
    if (-not $conda) { return $null }
    $python = Join-Path $MiniforgeRoot "envs\$EnvName\python.exe"
    if (Test-Path -LiteralPath $python) { return $python }
    return $null
}

function Require-EnvironmentPython {
    $python = Get-EnvironmentPython
    if (-not $python) {
        throw (Get-InstallerText 'missing_environment')
    }
    return $python
}

function Test-NvidiaDriver {
    $candidates = @(
        (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe')
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $candidates) { return $false }
    & $candidates[0] --query-gpu=name --format=csv,noheader 2>$null | Out-Null
    return $LASTEXITCODE -eq 0
}

function Test-Preflight {
    Write-Step (Get-InstallerText 'step_preflight')
    if (-not $IsWindows -and $PSVersionTable.PSVersion.Major -ge 6) {
        throw (Get-InstallerText 'windows_only')
    }
    if (-not (Test-Path -LiteralPath $EnvironmentFile)) {
        throw (Get-InstallerText 'missing_install_file' @($EnvironmentFile))
    }
    $driveName = ([System.IO.Path]::GetPathRoot($env:USERPROFILE)).TrimEnd('\').TrimEnd(':')
    $drive = Get-PSDrive -Name $driveName -ErrorAction SilentlyContinue
    if ($drive -and $drive.Free -lt 12GB) {
        throw (Get-InstallerText 'insufficient_space' @(($drive.Free / 1GB)))
    }
    Write-Host (Get-InstallerText 'windows_ok')
    Write-Host (Get-InstallerText $(if (Test-NvidiaDriver) { 'nvidia_detected' } else { 'nvidia_missing' }))
    Write-Host (Get-InstallerText 'work_folder' @($WorkRoot))
}

function Install-Miniforge {
    if (Find-Conda) { return }
    Write-Step (Get-InstallerText 'step_miniforge')
    $installer = Join-Path $env:TEMP 'HAGISILTA-Miniforge3-Windows-x86_64.exe'
    $url = 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe'
    Invoke-WebRequest -Uri $url -OutFile $installer -UseBasicParsing
    New-Item -ItemType Directory -Path $StateRoot -Force | Out-Null
    $arguments = "/InstallationType=JustMe /RegisterPython=0 /AddToPath=0 /S /D=$MiniforgeRoot"
    $process = Start-Process -FilePath $installer -ArgumentList $arguments -WindowStyle Hidden -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    if ($process.ExitCode -ne 0 -or -not (Find-Conda)) {
        throw (Get-InstallerText 'miniforge_failed' @($process.ExitCode))
    }
}

function Install-PythonEnvironment {
    $conda = Find-Conda
    if (-not $conda) { throw (Get-InstallerText 'conda_missing') }
    $python = Get-EnvironmentPython
    if ($python) {
        Write-Step (Get-InstallerText 'step_repair_python')
        & $conda env update -n $EnvName -f $EnvironmentFile --prune
    } else {
        Write-Step (Get-InstallerText 'step_install_python')
        & $conda env create -f $EnvironmentFile
    }
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'python_install_failed' @($LASTEXITCODE)) }
}

function Install-PyTorchBackend {
    $python = Require-EnvironmentPython
    if (Test-NvidiaDriver) {
        Write-Step (Get-InstallerText 'step_gpu')
        & $python -m pip install --upgrade 'torch==2.13.0+cu130' 'torchvision==0.28.0+cu130' `
            --index-url 'https://download.pytorch.org/whl/cu130'
    } else {
        Write-Step (Get-InstallerText 'step_cpu')
        & $python -m pip install --upgrade 'torch==2.13.0' 'torchvision==0.28.0' `
            --index-url 'https://download.pytorch.org/whl/cpu'
    }
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'torch_failed' @($LASTEXITCODE)) }
    & $python -m pip install 'numpy<2' 'filelock>=3.12,<3.16'
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'compatibility_failed') }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'dependency_conflict') }
}

function Set-WorkFolder {
    $python = Require-EnvironmentPython
    Write-Step (Get-InstallerText 'step_work_folder')
    & $python -X utf8 $RuntimeScript --set-work-root $WorkRoot --create
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'work_folder_failed') }
}

function Install-Shortcuts {
    Write-Step (Get-InstallerText 'step_shortcuts')
    $runFile = Join-Path $RepoRoot 'HAGISILTA-Run.cmd'
    # Keep a versioned icon filename so Windows does not retain an older
    # desktop/start-menu shortcut icon from its shell cache.
    $iconFile = Join-Path $RepoRoot 'assets\hagisilta-app.ico'
    $shell = New-Object -ComObject WScript.Shell
    $targets = @(
        (Join-Path ([Environment]::GetFolderPath('Desktop')) 'HAGISILTA.lnk'),
        (Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\HAGISILTA.lnk')
    )
    foreach ($target in $targets) {
        $shortcut = $shell.CreateShortcut($target)
        $shortcut.TargetPath = $runFile
        $shortcut.WorkingDirectory = $RepoRoot
        $shortcut.Description = Get-InstallerText 'shortcut_description'
        if (Test-Path -LiteralPath $iconFile) {
            $shortcut.IconLocation = "$iconFile,0"
        }
        $shortcut.Save()
    }
}

function Test-Environment {
    $python = Require-EnvironmentPython
    Write-Step (Get-InstallerText 'step_verify')
    & $python -X utf8 (Join-Path $PSScriptRoot 'hagisilta_environment_check.py')
    if ($LASTEXITCODE -ne 0) { throw (Get-InstallerText 'verify_failed') }
}

function Start-Hagisilta {
    $python = Require-EnvironmentPython
    if (-not (Test-Path -LiteralPath $AppScript)) { throw (Get-InstallerText 'missing_app_file' @($AppScript)) }
    $launchTarget = if (Test-Path -LiteralPath $LauncherScript) { $LauncherScript } else { $AppScript }
    $pythonw = Join-Path (Split-Path -Parent $python) 'pythonw.exe'
    if (-not (Test-Path -LiteralPath $pythonw)) { $pythonw = $python }
    Start-Process -FilePath $pythonw -ArgumentList @('-X', 'utf8', "`"$launchTarget`"") -WorkingDirectory $RepoRoot
    Write-Host (Get-InstallerText 'app_started') -ForegroundColor Green
}

function Invoke-Install([bool]$LaunchAfter) {
    Test-Preflight
    Install-Miniforge
    Install-PythonEnvironment
    Install-PyTorchBackend
    Set-WorkFolder
    Save-InstallerLanguage
    Test-Environment
    Install-Shortcuts
    Write-Host ("`n" + (Get-InstallerText 'installation_completed')) -ForegroundColor Green
    if ($LaunchAfter) { Start-Hagisilta }
}

function Invoke-Logged([scriptblock]$Operation) {
    New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
    $logFile = Join-Path $LogRoot ("install-{0}.log" -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    try {
        Start-Transcript -LiteralPath $logFile -Force | Out-Null
        & $Operation
    } catch {
        Write-Host ("`n" + (Get-InstallerText 'error' @($_.Exception.Message))) -ForegroundColor Red
        Write-Host (Get-InstallerText 'installation_log' @($logFile)) -ForegroundColor Yellow
        throw
    } finally {
        try { Stop-Transcript | Out-Null } catch {}
    }
}

function Show-Menu {
    while ($true) {
        Clear-Host
        Write-Host (Get-InstallerText 'title_Menu') -ForegroundColor Green
        Write-Host (Get-InstallerText 'menu_run')
        Write-Host (Get-InstallerText 'menu_install')
        Write-Host (Get-InstallerText 'menu_repair')
        Write-Host (Get-InstallerText 'menu_verify')
        Write-Host (Get-InstallerText 'menu_folder')
        Write-Host (Get-InstallerText 'menu_exit')
        $choice = Read-Host (Get-InstallerText 'select')
        try {
            switch ($choice) {
                '1' { Start-Hagisilta; return }
                '2' { Invoke-Logged { Invoke-Install $true }; return }
                '3' { Invoke-Logged { Invoke-Install $false } }
                '4' { Test-Environment }
                '5' {
                    $python = Require-EnvironmentPython
                    $resolved = & $python -X utf8 $RuntimeScript --create
                    Start-Process explorer.exe -ArgumentList "`"$($resolved | Select-Object -Last 1)`""
                }
                '0' { return }
                default { Write-Host (Get-InstallerText 'select_number') -ForegroundColor Yellow }
            }
        } catch {
            Write-Host (Get-InstallerText 'error' @($_.Exception.Message)) -ForegroundColor Red
        }
        Read-Host (Get-InstallerText 'return_menu') | Out-Null
    }
}

function Invoke-InstallerMain {
    $previousLanguage = $env:HAGISILTA_UI_LANGUAGE
    $script:InstallerExitCode = 0
    try {
        Initialize-InstallerLanguage
        # Repair must remain reachable even when the existing Python is broken.
        # Install/Repair save after rebuilding the environment in Invoke-Install.
        if ($Action -in @('Run', 'Verify')) {
            Save-InstallerLanguage
        } elseif ($Action -eq 'Menu') {
            try { Save-InstallerLanguage } catch {
                Write-Host (Get-InstallerText 'language_save_deferred') -ForegroundColor Yellow
            }
        }
        switch ($Action) {
            'Install' { Invoke-Logged { Invoke-Install $true } }
            'Repair' { Invoke-Logged { Invoke-Install $false } }
            'Verify' { Test-Environment }
            'Run' { Start-Hagisilta }
            'Menu' { Show-Menu }
        }
    } catch {
        $script:InstallerExitCode = 1
        if ($script:InstallerLanguage -in @('ko', 'en') -and $script:InstallerCatalogs) {
            Write-Host (Get-InstallerText 'error' @($_.Exception.Message)) -ForegroundColor Red
            Write-Host (Get-InstallerText 'exit_code' @($script:InstallerExitCode))
        } else {
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
    } finally {
        try {
            if ($WaitForKey -and $script:InstallerLanguage -in @('ko', 'en') -and $script:InstallerCatalogs) {
                Read-Host (Get-InstallerText 'close_window') | Out-Null
            }
        } finally {
            $env:HAGISILTA_UI_LANGUAGE = $previousLanguage
        }
    }
}

# Dot-sourcing exposes functions for isolated tests without installing or launching.
if ($MyInvocation.InvocationName -ne '.') {
    Invoke-InstallerMain
    exit $script:InstallerExitCode
}
