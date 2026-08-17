@echo off
rem Copyright (c) KMG. All Rights Reserved.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem ##

where powershell.exe >nul 2>nul
if errorlevel 1 (
    echo sbk-charts: ERROR: PowerShell is required to bootstrap sbk-charts. 1>&2
    exit /b 1
)

powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0sbk-charts.ps1" %*
exit /b %ERRORLEVEL%
