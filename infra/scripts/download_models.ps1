$ErrorActionPreference = "Continue"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ModelsDir = Join-Path $ScriptDir "..\..\models_cache"

if (-not (Test-Path -Path $ModelsDir)) {
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
}

Write-Host "Downloading LivePortrait..."
$LivePortraitDir = Join-Path $ModelsDir "liveportrait"
if (-not (Test-Path -Path $LivePortraitDir)) {
    git clone https://github.com/KwaiVGI/LivePortrait $LivePortraitDir
}
try { pip install -r (Join-Path $LivePortraitDir "requirements.txt") } catch { Write-Host "Skipping LivePortrait strict pip install." }
hf download KwaiVGI/LivePortrait --local-dir (Join-Path $LivePortraitDir "weights") --exclude "*.git*"

Write-Host "Downloading MuseTalk..."
$MuseTalkDir = Join-Path $ModelsDir "musetalk"
if (-not (Test-Path -Path $MuseTalkDir)) {
    git clone https://github.com/TMElyralab/MuseTalk $MuseTalkDir
}
try { pip install -r (Join-Path $MuseTalkDir "requirements.txt") } catch { Write-Host "Skipping MuseTalk strict pip install." }
hf download TMElyralab/MuseTalk --local-dir (Join-Path $MuseTalkDir "weights") --exclude "*.git*"

Write-Host "Downloading Wav2Lip..."
$Wav2LipDir = Join-Path $ModelsDir "wav2lip"
if (-not (Test-Path -Path $Wav2LipDir)) {
    git clone https://github.com/Rudrabha/Wav2Lip $Wav2LipDir
}
try { pip install -r (Join-Path $Wav2LipDir "requirements.txt") } catch { Write-Host "Skipping Wav2Lip strict pip install." }

Write-Host "Downloading GFPGAN..."
$GFPGANDir = Join-Path $ModelsDir "gfpgan"
if (-not (Test-Path -Path $GFPGANDir)) {
    git clone https://github.com/TencentARC/GFPGAN $GFPGANDir
}
try { pip install -r (Join-Path $GFPGANDir "requirements.txt") } catch { Write-Host "Skipping GFPGAN strict pip install." }

$GFPGANPretrainedDir = Join-Path $GFPGANDir "experiments\pretrained_models"
if (-not (Test-Path -Path $GFPGANPretrainedDir)) {
    New-Item -ItemType Directory -Force -Path $GFPGANPretrainedDir | Out-Null
}
$GFPGANModelPath = Join-Path $GFPGANPretrainedDir "GFPGANv1.4.pth"
if (-not (Test-Path -Path $GFPGANModelPath)) {
    try {
        Invoke-WebRequest -Uri "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth" -OutFile $GFPGANModelPath
    } catch {
        Write-Host "Note: Invoke-WebRequest failed, please download GFPGANv1.4 manually."
    }
}

Write-Host "Downloading RIFE..."
$RIFEDir = Join-Path $ModelsDir "rife"
if (-not (Test-Path -Path $RIFEDir)) {
    git clone https://github.com/hzwer/ECCV2022-RIFE $RIFEDir
}
try { pip install -r (Join-Path $RIFEDir "requirements.txt") } catch { Write-Host "Skipping RIFE strict pip install." }

Write-Host "TODO: clone/download other chosen models into $ModelsDir"
Write-Host "  - InsightFace:   pip install insightface (downloads weights on first use)"
