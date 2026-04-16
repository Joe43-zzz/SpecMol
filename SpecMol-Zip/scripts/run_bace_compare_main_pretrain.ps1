$repoRoot = Split-Path -Parent $PSScriptRoot
$pythonExe = Join-Path (Split-Path -Parent $repoRoot) "python38-embed\python.exe"
$logDir = Join-Path $repoRoot "results\main_pretrain_compare"

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Set-Location $repoRoot

$baselineLog = Join-Path $logDir "baseline.log"
$pairLog = Join-Path $logDir "pair.log"

Write-Host "[compare] repoRoot=$repoRoot"
Write-Host "[compare] pythonExe=$pythonExe"
Write-Host "[compare] baselineLog=$baselineLog"
Write-Host "[compare] pairLog=$pairLog"

& $pythonExe -u "main_pretrain.py" --task bace --path "down_task_2d" --epochs 200 --gpu -1 2>&1 |
    Tee-Object -FilePath $baselineLog

& $pythonExe -u "main_pretrain.py" --task bace --path "down_task/processed" --epochs 200 --gpu -1 2>&1 |
    Tee-Object -FilePath $pairLog
