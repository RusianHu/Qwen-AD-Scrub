# Qwen-AD-Scrub 启动脚本
# PowerShell 脚本

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Qwen-AD-Scrub 启动脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Python
Write-Host "[1/3] 检查 Python 环境..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Python 已安装: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "✗ Python 未安装或未添加到 PATH" -ForegroundColor Red
    Write-Host "请安装 Python 3.9 或更高版本" -ForegroundColor Red
    pause
    exit 1
}

# 检查依赖
Write-Host ""
Write-Host "[2/3] 检查依赖..." -ForegroundColor Yellow

$requirementsFile = "requirements.txt"
if (Test-Path $requirementsFile) {
    Write-Host "发现 requirements.txt" -ForegroundColor Green
    
    $choice = Read-Host "是否安装/更新依赖? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        Write-Host "正在安装依赖..." -ForegroundColor Yellow
        python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ 依赖安装完成" -ForegroundColor Green
        } else {
            Write-Host "✗ 依赖安装失败" -ForegroundColor Red
        }
    }
} else {
    Write-Host "✗ 未找到 requirements.txt" -ForegroundColor Red
}

# 检查模型和 FFmpeg
Write-Host ""
Write-Host "[3/3] 检查模型和 FFmpeg..." -ForegroundColor Yellow

$setupScript = "setup_dependencies.py"
if (Test-Path $setupScript) {
    $choice = Read-Host "是否运行依赖检查脚本? (y/n)"
    if ($choice -eq "y" -or $choice -eq "Y") {
        python $setupScript
    }
}

# 启动应用
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  启动 Web UI" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$appScript = "app.py"
if (Test-Path $appScript) {
    Write-Host "正在启动应用..." -ForegroundColor Yellow
    Write-Host "浏览器将自动打开 http://localhost:7860" -ForegroundColor Green
    Write-Host ""
    Write-Host "按 Ctrl+C 停止服务器" -ForegroundColor Yellow
    Write-Host ""
    
    python $appScript
} else {
    Write-Host "✗ 未找到 app.py" -ForegroundColor Red
    pause
    exit 1
}

