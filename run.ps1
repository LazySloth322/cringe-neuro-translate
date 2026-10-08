param(
    [Parameter(Mandatory = $true)]
    [string]$Video
)

$ErrorActionPreference = "Stop"

# ============================================================
# PIPELINE ROOT
# ============================================================
# This is ALWAYS the directory where run_pipeline.ps1 lives.
$PipelineDir = $PSScriptRoot

# ============================================================
# STAGE DIRECTORIES
# ============================================================
$STTDir       = Join-Path $PipelineDir "speechtotext"
$TranslateDir = Join-Path $PipelineDir "summarize"
$TTSDir       = Join-Path $PipelineDir "tts"

# ============================================================
# PYTHON EXECUTABLES
# ============================================================
# Each stage gets its OWN Python environment.
$PythonSTT       = Join-Path $STTDir ".venv\Scripts\python.exe"
$PythonTranslate = Join-Path $TranslateDir ".venv\Scripts\python.exe"
$PythonTTS       = Join-Path $TTSDir ".venv\Scripts\python.exe"

# ============================================================
# VALIDATE PYTHON ENVIRONMENTS
# ============================================================
if (-not (Test-Path $PythonSTT)) {
    throw "STT Python environment not found: $PythonSTT"
}

if (-not (Test-Path $PythonTranslate)) {
    throw "Translation Python environment not found: $PythonTranslate"
}

if (-not (Test-Path $PythonTTS)) {
    throw "TTS Python environment not found: $PythonTTS"
}

# ============================================================
# INPUT VIDEO
# ============================================================
try {
    $VideoPath = (Resolve-Path $Video -ErrorAction Stop).Path
}
catch {
    throw "Input video not found: $Video"
}

# Filename without extension
$Name = [System.IO.Path]::GetFileNameWithoutExtension($VideoPath)

# ============================================================
# OUTPUT DIRECTORY
# ============================================================
# IMPORTANT:
# This is created NEXT TO run_pipeline.ps1.
#
# Example:
# L:\python_translate\run_pipeline.ps1
# + L:\python_translate\movie.mp4
# =>
# L:\python_translate\movie_output\
# ============================================================
$OutputDir = Join-Path $PipelineDir "${Name}_output"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# ============================================================
# FILE PATHS
# ============================================================
$InputSrt = Join-Path $OutputDir "output.srt"
$CompressedSrt = Join-Path $OutputDir "${Name}_ru_compressed.srt"
$FinalVideo = Join-Path $OutputDir "${Name}_ru_dub_mixed_subtitles.mp4"

# ============================================================
# SHOW CONFIGURATION
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " PIPELINE"
Write-Host "============================================================"
Write-Host "Pipeline directory : $PipelineDir"
Write-Host "Input video        : $VideoPath"
Write-Host "Output directory   : $OutputDir"
Write-Host ""
Write-Host "STT Python         : $PythonSTT"
Write-Host "Translate Python   : $PythonTranslate"
Write-Host "TTS Python         : $PythonTTS"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# STAGE 1 -" SPEECH TO TEXT
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " STAGE 1/3 - STT"
Write-Host "============================================================"
Write-Host ""

Push-Location $STTDir

try {
    & $PythonSTT ".\speaker_pipeline.py" $VideoPath

    if ($LASTEXITCODE -ne 0) {
        throw "STT stage failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

# Make sure STT actually produced the expected file
if (-not (Test-Path $InputSrt)) {
    throw "STT stage finished, but expected file was not found: $InputSrt"
}

Write-Host ""
Write-Host "STT completed successfully."
Write-Host "Created: $InputSrt"


# ============================================================
# STAGE 2 -" TRANSLATION + COMPRESSION
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " STAGE 2/3 - TRANSLATION"
Write-Host "============================================================"
Write-Host ""

Push-Location $TranslateDir

try {
	& $PythonTranslate `
		(Join-Path $TranslateDir "srt_semantic_compressor_v2.py") `
		$InputSrt `
		"-o" `
		$CompressedSrt `
		"--translate-and-compress"

    if ($LASTEXITCODE -ne 0) {
        throw "Translation stage failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

# Make sure translation produced the expected file
if (-not (Test-Path $CompressedSrt)) {
    throw "Translation stage finished, but expected file was not found: $CompressedSrt"
}

Write-Host ""
Write-Host "Translation completed successfully."
Write-Host "Created: $CompressedSrt"


# ============================================================
# STAGE 3 -" TTS / DUBBING
# ============================================================
Write-Host ""
Write-Host "============================================================"
Write-Host " STAGE 3/3 - TTS / DUBBING"
Write-Host "============================================================"
Write-Host ""

Push-Location $TTSDir

try {
    & $PythonTTS `
        ".\chatterbox_dubber_v3.py" `
        $CompressedSrt `
        "--audio" `
        $VideoPath `
        "--video" `
        $VideoPath `
        "--mix-original"

    if ($LASTEXITCODE -ne 0) {
        throw "TTS stage failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "TTS completed successfully."

# ============================================================
# DONE
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host " PIPELINE COMPLETED SUCCESSFULLY"
Write-Host "============================================================"
Write-Host ""
Write-Host "Final video: "
Write-Host $FinalVideo
Write-Host ""
