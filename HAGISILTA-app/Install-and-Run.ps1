# SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
# SPDX-License-Identifier: Apache-2.0
[CmdletBinding()]
param(
    [string]$WorkRoot = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'HAGISILTA-work'),
    [switch]$SkipInstall,
    [ValidateSet('auto', 'ko', 'en')]
    [string]$Language = 'auto'
)

$ErrorActionPreference = 'Stop'
$installer = Join-Path $PSScriptRoot 'scripts\HAGISILTA-Installer.ps1'
if ($SkipInstall) {
    & $installer -Action Run -WorkRoot $WorkRoot -Language $Language
} else {
    & $installer -Action Install -WorkRoot $WorkRoot -Language $Language
}
exit $LASTEXITCODE
