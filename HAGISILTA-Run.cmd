@echo off
REM SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
REM SPDX-License-Identifier: Apache-2.0
chcp 65001 >nul
powershell.exe -NoLogo -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "%~dp0scripts\HAGISILTA-Installer.ps1" -Action Run %*
if errorlevel 1 (
  powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\HAGISILTA-Installer.ps1" -Action Run -WaitForKey %*
)
