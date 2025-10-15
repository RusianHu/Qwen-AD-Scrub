# 🎬 Qwen-AD-Scrub

基于 **Qwen3-VL 系列模型** 和 **FFmpeg** 的智能视频广告去除工具

<img width="1381" height="1188" alt="image" src="https://github.com/user-attachments/assets/20b7f5c2-55a5-4419-bd18-de40e8c8f976" />

<img width="997" height="346" alt="image" src="https://github.com/user-attachments/assets/c58a10ba-e763-4f42-adc9-9a7147ed2270" />

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA GPU (推荐 RTX 3060 或更高)
  - 4B 模型: 8GB+ 显存 (8-bit 量化)
  - 8B 模型: 12GB+ 显存 (8-bit 量化)
  - 30B 模型: 16GB+ 显存 (8-bit 量化)
- **CPU**: 现代多核处理器
- **内存**: 建议 16GB 以上
- **存储**: 至少 30GB 可用空间（每个模型 8-30GB）

### 软件要求
- **操作系统**: Windows 11 64位
- **Python**: 3.9 或更高版本
- **CUDA**: 11.8 或更高版本 (用于 GPU 加速)

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/RusianHu/Qwen-AD-Scrub.git
cd Qwen-AD-Scrub
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型和 FFmpeg

运行依赖检查脚本：

```bash
python setup_dependencies.py
```

这个脚本提供交互式菜单，可以：
- 下载任意支持的 Qwen3-VL 模型（4B、8B、30B 等）
- 检查已安装的模型
- 从 HF-Mirror 镜像站下载模型（自动断点续传）
- 检查并解压 FFmpeg

**支持的模型：**
- **Qwen3-VL-4B-Instruct** (推荐): 平衡性能与速度，适合大多数场景
- **Qwen3-VL-8B-Instruct**: 更强的理解能力，适合复杂场景
- **Qwen3-VL-30B-A3B-Instruct**: 最强性能，需要高端显卡

### 4. 启动 Web UI

**推荐方式**（自动优化配置）：

```bash
python start_optimized.py
```

或使用标准方式：

```bash
python app.py
```

或使用 PowerShell 脚本：

```powershell
.\run.ps1
```

然后在浏览器中打开 `http://localhost:7860`

## 📖 使用指南

### 基本流程

1. **初始化系统**
   - 在顶部的"系统初始化与模型管理"区域
   - 点击"🚀 初始化系统"按钮
   - 系统会自动扫描并显示可用模型
   - 等待初始化完成

2. **选择并加载模型**
   - 从"选择模型"下拉框中选择要使用的模型
   - **8-bit 量化默认已启用**（节省 50% 显存，推荐保持）
   - 可选：勾选"使用 Flash Attention 2"（需要 RTX 30/40/50 系列 GPU）
   - 点击"📦 加载模型"按钮
   - 等待加载完成（首次可能需要几分钟）

3. **切换模型（可选）**
   - 如需使用不同的模型，从下拉框中选择新模型
   - 点击"🔄 切换模型"按钮
   - 系统会自动卸载当前模型并加载新模型

4. **分析视频**
   - 切换到"视频分析"标签页
   - 上传要处理的视频文件
   - 调整采样帧率（默认 0.3 fps，长视频建议 0.2-0.3 fps）
   - 可选：输入自定义提示词
   - 点击"开始分析"

5. **去除广告**
   - 分析完成后，切换到"视频处理"标签页
   - 查看检测到的广告片段
   - 点击"去除广告并导出"
   - 等待处理完成，下载处理后的视频

### 参数调优建议

#### 采样帧率 (fps)

| 视频时长 | 推荐 fps | 显存占用 | 精度 | 适用场景 |
|---------|---------|---------|------|---------|
| < 1分钟 | 0.5-1.0 | 中等 | 高 | 短视频、精细检测 |
| 1-3分钟 | 0.3-0.5 | 较低 | 中高 | 中等视频、平衡模式 |
| 3-10分钟 | 0.2-0.3 | 低 | 中 | 长视频、快速检测 |
| > 10分钟 | 0.2 | 最低 | 中低 | 超长视频、极限节省 |

#### 自定义提示词示例

```
请识别视频中的所有广告片段，包括：
1. 品牌推广和产品展示
2. 赞助商信息
3. 片头片尾广告
4. 中插广告

请给出每个广告的准确时间范围（格式：开始秒数-结束秒数）。
```

## 🔧 高级配置

### 模型配置

模型默认存放在 `models/Qwen3-VL-4B-Instruct/` 目录下。

**自定义模型路径**:
```python
from src.model_loader import ModelLoader

model_loader = ModelLoader(model_path="/path/to/your/model")
model_loader.load_model(use_8bit=True)  # 推荐启用 8-bit 量化
```

### FFmpeg 配置

FFmpeg 默认存放在 `FFmpeg/bin/` 目录下。

**使用系统 FFmpeg**:
```python
from src.video_processor import VideoProcessor

video_processor = VideoProcessor(ffmpeg_path="/usr/bin/ffmpeg")  # Linux
# 或
video_processor = VideoProcessor(ffmpeg_path="C:/ffmpeg/bin/ffmpeg.exe")  # Windows
```

### 显存优化配置

**8-bit 量化参数**:
```python
# 在 src/model_loader.py 中
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,              # 启用 8-bit 量化
    llm_int8_threshold=6.0,         # 异常值阈值
    llm_int8_has_fp16_weight=False  # 不保留 FP16 权重（节省更多显存）
)
```

**视频处理参数**:
```python
# 在 src/ad_detector.py 中
result = detector.detect_ads_in_video(
    video_path="video.mp4",
    fps=0.3,                    # 采样帧率（越低越省显存）
    min_pixels=4 * 28 * 28,     # 最小像素数
    max_pixels=64 * 28 * 28,    # 最大像素数（越低越省显存）
)
```

## 📝 技术栈

### 核心技术
- **AI 模型**: Qwen3-VL-4B-Instruct (Qwen2VLForConditionalGeneration)
- **深度学习框架**: PyTorch 2.0+, Transformers 4.37+
- **量化技术**: bitsandbytes (8-bit 量化)
- **视频处理**: FFmpeg, qwen-vl-utils, decord
- **Web 框架**: Gradio 4.0+

### 辅助库
- **数据处理**: NumPy, Pillow
- **工具**: tqdm, requests, pathlib

## 💡 Python API 使用示例

```python
from src.model_loader import ModelLoader
from src.video_processor import VideoProcessor
from src.ad_detector import AdDetector

# 1. 初始化并加载模型
model_loader = ModelLoader()
model_loader.load_model(use_8bit=True)  # 启用 8-bit 量化

# 2. 创建检测器
detector = AdDetector(model_loader)

# 3. 检测广告
result = detector.detect_ads_in_video(
    video_path="video.mp4",
    fps=0.3,  # 采样帧率
    custom_prompt="请识别视频中的所有广告片段"
)

# 4. 查看结果
print(f"是否有广告: {result['has_ads']}")
print(f"广告片段: {result['ad_segments']}")
print(f"模型输出: {result['raw_output']}")

# 5. 去除广告（可选）
if result['has_ads']:
    video_processor = VideoProcessor()
    # 计算保留片段...
    # video_processor.cut_video(...)
```

更多示例请查看 `examples/example_usage.py`

## 🔄 更新日志

### v1.0.0 (2025-10-15)
- ✅ 完整的视频广告检测功能
- ✅ 8-bit 量化默认启用，节省 50% 显存
- ✅ 优化的 Web UI 界面
- ✅ 完善的中文路径支持
- ✅ 自动显存管理和多卡支持
- ✅ 详细的文档和示例代码

## 🙏 致谢

- [Qwen Team](https://github.com/QwenLM) - 提供优秀的 Qwen3-VL 视觉语言模型
- [HuggingFace](https://huggingface.co/) - Transformers 库和模型托管
- [FFmpeg](https://ffmpeg.org/) - 强大的视频处理工具
- [Gradio](https://gradio.app/) - 简洁易用的 Web UI 框架
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) - 高效的量化库

## 📧 联系方式

- **GitHub**: [@RusianHu](https://github.com/RusianHu)
- **Email**: hu_bo_cheng@qq.com
- **Issues**: [GitHub Issues](https://github.com/RusianHu/Qwen-AD-Scrub/issues)

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
