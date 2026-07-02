$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$PlatformTools = Join-Path $ProjectRoot "tools\platform-tools"
$ScrcpyTools = Join-Path $ProjectRoot "tools\scrcpy-win64-v4.0"

$env:Path = "$PlatformTools;$ScrcpyTools;$env:Path"

Write-Output "Phase 1 tools added to PATH for this terminal:"
Write-Output "  $PlatformTools"
Write-Output "  $ScrcpyTools"
Write-Output ""
Write-Output "Use: conda activate aiml_cuda"

