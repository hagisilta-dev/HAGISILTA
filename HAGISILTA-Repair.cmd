@echo off
REM SPDX-FileCopyrightText: 2026 Ok Seungwon and Song Jun-Hwi
REM SPDX-License-Identifier: Apache-2.0
chcp 65001 >nul
title HAGISILTA
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\HAGISILTA-Installer.ps1" -Action Repair -WaitForKey %*
set "EXIT_CODE=%ERRORLEVEL%"
exit /b %EXIT_CODE%
