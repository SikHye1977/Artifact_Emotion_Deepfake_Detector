param(
    [string]$Output = "reports/artifact_emotion_gated_optimized_v1",
    [string]$EncoderDevice = "cuda:0", [string]$HeadDevice = "cpu",
    [int]$Epochs = 30, [int]$Seed = 42, [int]$FaceBatchSize = 1,
    [string]$LegacyVideoTrain = "", [switch]$Resume
)
$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path
$pythonExe = Join-Path $projectRoot ".venv/Scripts/python.exe"
$priorPath = $env:PATH
$priorUtf8 = $env:PYTHONUTF8
$priorWorkspaceConfig = $env:CUBLAS_WORKSPACE_CONFIG
$env:PATH = "$(Join-Path $projectRoot '.venv/Scripts');$priorPath"
$env:PYTHONUTF8 = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Push-Location $projectRoot
try {
    $arguments = @((Join-Path $PSScriptRoot 'pipeline.py'), '--output', $Output, '--encoder-device', $EncoderDevice,
                  '--head-device', $HeadDevice, '--epochs', $Epochs, '--seed', $Seed, '--face-batch-size', $FaceBatchSize)
    if ($Resume) { $arguments += '--resume' }
    if ($LegacyVideoTrain) { $arguments += @('--legacy-video-train', $LegacyVideoTrain) }
    & $pythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "Pipeline stopped. Fix the cause and repeat the same command with -Resume." }
} finally {
    Pop-Location
    $env:PATH = $priorPath
    $env:PYTHONUTF8 = $priorUtf8
    $env:CUBLAS_WORKSPACE_CONFIG = $priorWorkspaceConfig
}
