@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
for %%I in ("%REPO_ROOT%\..") do set "WORKSPACE_ROOT=%%~fI"
set "PYTHON_EXE=%WORKSPACE_ROOT%\python38-embed\python.exe"
set "LOG_DIR=%REPO_ROOT%\results\main_pretrain_compare"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"

cd /d "%REPO_ROOT%"

echo [compare] repoRoot=%REPO_ROOT%
echo [compare] pythonExe=%PYTHON_EXE%
echo [compare] logDir=%LOG_DIR%

"%PYTHON_EXE%" -u main_pretrain.py --task bace --path down_task_2d --epochs 200 --gpu -1 > "%LOG_DIR%\baseline.log" 2>&1
"%PYTHON_EXE%" -u main_pretrain.py --task bace --path down_task/processed --epochs 200 --gpu -1 > "%LOG_DIR%\pair.log" 2>&1

endlocal
