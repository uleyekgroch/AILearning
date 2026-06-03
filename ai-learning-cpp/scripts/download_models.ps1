#Requires -Version 5.1
<#
.SYNOPSIS
    自动下载推荐的 llama.cpp 预训练模型 (Windows PowerShell)

.DESCRIPTION
    下载 bge-small-zh-v1.5、Qwen2.5-3B-Instruct GGUF 和 CLIP ONNX 模型文件。
    支持断点续传和镜像回退。

.PARAMETER Dir
    模型下载目录 (默认: ../models)

.PARAMETER Model
    仅下载指定类型: embedding | llm | clip | all (默认: all)

.EXAMPLE
    .\download_models.ps1
    .\download_models.ps1 -Model embedding
    .\download_models.ps1 -Model clip
    .\download_models.ps1 -Dir "C:\Models"
#>
[CmdletBinding()]
param(
    [string]$Dir = (Join-Path $PSScriptRoot "..\models"),
    [ValidateSet("all", "embedding", "llm", "clip")]
    [string]$Model = "all"
)

# ── 配置 ─────────────────────────────────────────────────────────
$Models = @{
    embedding = @{
        Name = "bge-small-zh-v1.5 Q4_K_M"
        Purpose = "Chinese text embedding"
        Size = "~15 MB"
        PrimaryUrl = "https://hf-mirror.com/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-q4_k_m.gguf"
        FallbackUrl = "https://huggingface.co/CompendiumLabs/bge-small-zh-v1.5-gguf/resolve/main/bge-small-zh-v1.5-q4_k_m.gguf"
        FileName = "bge-small-zh-v1.5-q4_k_m.gguf"
    }
    llm = @{
        Name = "Qwen2.5-3B-Instruct Q4_K_M"
        Purpose = "Chinese dialogue & reasoning"
        Size = "~2.0 GB"
        PrimaryUrl = "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        FallbackUrl = "https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
        FileName = "qwen2.5-3b-instruct-q4_k_m.gguf"
    }
    clip = @{
        Name = "CLIP ViT-B/32 Image Encoder ONNX"
        Purpose = "Vision encoding for multimodal perception"
        Size = "~330 MB"
        PrimaryUrl = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/model.onnx"
        FallbackUrl = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/onnx/model.onnx"
        FileName = "clip-vit-base-patch32.onnx"
    }
}

# ── 辅助函数 ──────────────────────────────────────────────────────
function Write-Info  { param([string]$msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-OK    { param([string]$msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warn  { param([string]$msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$msg) Write-Host "[ERR] $msg" -ForegroundColor Red }

function Download-File {
    param(
        [string]$Url,
        [string]$OutputPath,
        [string]$Desc
    )

    Write-Info "Downloading ${Desc}..."
    Write-Info "  URL: ${Url}"
    Write-Info "  Dest: ${OutputPath}"

    # 检查已存在
    if (Test-Path $OutputPath) {
        $existingSize = (Get-Item $OutputPath).Length
        if ($existingSize -gt 1MB) {
            Write-Warn "File already exists ($existingSize bytes). Skipping."
            Write-Warn "Remove-Item '${OutputPath}' to force re-download."
            return $true
        }
    }

    try {
        # 使用 BITS 传输（Windows 原生，支持断点续传）
        $startTime = Get-Date
        Start-BitsTransfer -Source $Url -Destination $OutputPath `
                           -DisplayName "Downloading ${Desc}" `
                           -Description $Desc `
                           -ErrorAction Stop
        $duration = (Get-Date) - $startTime
        $fileSize = (Get-Item $OutputPath).Length
        Write-OK "Downloaded: ${OutputPath} (${fileSize} bytes) in ${duration:%m}m ${duration:%s}s"
        return $true
    }
    catch {
        # 如果 BITS 失败，尝试 .NET WebClient
        Write-Warn "BITS transfer failed: $($_.Exception.Message)"
        Write-Info "Falling back to WebClient..."
        try {
            $wc = New-Object System.Net.WebClient
            $wc.DownloadFile($Url, $OutputPath)
            $fileSize = (Get-Item $OutputPath).Length
            Write-OK "Downloaded: ${OutputPath} (${fileSize} bytes)"
            return $true
        }
        catch {
            Write-Err "Download failed: $($_.Exception.Message)"
            if (Test-Path $OutputPath) { Remove-Item $OutputPath -Force }
            return $false
        }
    }
}

function Download-WithFallback {
    param(
        [hashtable]$ModelInfo
    )

    $outputPath = Join-Path $Dir $ModelInfo.FileName

    # 尝试主 URL
    if (Download-File -Url $ModelInfo.PrimaryUrl -OutputPath $outputPath -Desc $ModelInfo.Name) {
        return $true
    }

    # 尝试备用 URL
    Write-Warn "Primary mirror failed. Trying fallback..."
    if (Download-File -Url $ModelInfo.FallbackUrl -OutputPath $outputPath -Desc "$($ModelInfo.Name) (fallback)") {
        return $true
    }

    Write-Err "All mirrors failed for $($ModelInfo.Name)"
    return $false
}

# ── 主逻辑 ────────────────────────────────────────────────────────
New-Item -ItemType Directory -Force -Path $Dir | Out-Null
Write-Info "Model directory: $(Resolve-Path $Dir)"
Write-Host ""

$failed = 0

if ($Model -eq "all" -or $Model -eq "embedding") {
    Write-Info "========================================"
    Write-Info "Model: $($Models.embedding.Name)"
    Write-Info "Purpose: $($Models.embedding.Purpose) ($($Models.embedding.Size))"
    Write-Info "========================================"
    if (-not (Download-WithFallback $Models.embedding)) { $failed++ }
    Write-Host ""
}

if ($Model -eq "all" -or $Model -eq "llm") {
    Write-Info "========================================"
    Write-Info "Model: $($Models.llm.Name)"
    Write-Info "Purpose: $($Models.llm.Purpose) ($($Models.llm.Size))"
    Write-Info "========================================"
    Write-Warn "This is a large file (~2GB). Download may take several minutes."
    if (-not (Download-WithFallback $Models.llm)) { $failed++ }
    Write-Host ""
}

if ($Model -eq "all" -or $Model -eq "clip") {
    Write-Info "========================================"
    Write-Info "Model: $($Models.clip.Name)"
    Write-Info "Purpose: $($Models.clip.Purpose) ($($Models.clip.Size))"
    Write-Info "========================================"
    Write-Warn "Requires: cmake -DAI_LEARNING_WITH_ONNX=ON"
    Write-Warn "This is a large file (~330MB). Download may take a few minutes."
    if (-not (Download-WithFallback $Models.clip)) { $failed++ }
    Write-Host ""
}

# ── 摘要 ────────────────────────────────────────────────────────
Write-Host ""
Write-Info "========================================"
Write-Info "Download Summary"
Write-Info "========================================"

Get-ChildItem $Dir -Filter "*.gguf" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-OK "$($_.Name): $([math]::Round($_.Length / 1MB, 2)) MB"
}
Get-ChildItem $Dir -Filter "*.onnx" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-OK "$($_.Name): $([math]::Round($_.Length / 1MB, 2)) MB"
}

Write-Host ""
if ($failed -eq 0) {
    Write-OK "All requested models downloaded successfully!"
    Write-Host ""
    Write-Info "Next steps:"
    Write-Host "  .\verify_llama_cpp.exe $Dir\qwen2.5-3b-instruct-q4_k_m.gguf"
    Write-Host "  .\ai_learning_server.exe --llm-model $Dir\qwen2.5-3b-instruct-q4_k_m.gguf"
    if (Test-Path (Join-Path $Dir "clip-vit-base-patch32.onnx")) {
        Write-Host "  cmake -DAI_LEARNING_WITH_ONNX=ON .. && cmake --build ."
        Write-Host "  .\ai_learning_server.exe  # auto-detects CLIP model"
    }
    exit 0
} else {
    Write-Err "$failed download(s) failed."
    exit 1
}
