$Emulator = Join-Path $env:LOCALAPPDATA "Android\Sdk\emulator\emulator.exe"

if (-not (Test-Path -LiteralPath $Emulator)) {
    throw "Android emulator not found at $Emulator"
}

& $Emulator -avd Pixel_7_Playstore

