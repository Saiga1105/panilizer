$ErrorActionPreference = "Stop"

$nodePath = "C:\Program Files\nodejs"
if (Test-Path $nodePath) {
    $env:Path = "$nodePath;$env:Path"
}

Set-Location (Join-Path $PSScriptRoot "..\ui")
npm run dev
