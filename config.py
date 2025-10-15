"""
配置文件
"""

from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 支持的模型配置列表
SUPPORTED_MODELS = {
    'Qwen3-VL-4B-Instruct': {
        'display_name': 'Qwen3-VL-4B-Instruct (推荐)',
        'model_path': PROJECT_ROOT / 'models' / 'Qwen3-VL-4B-Instruct',
        'hf_mirror': 'https://hf-mirror.com',
        'hf_model_id': 'Qwen/Qwen3-VL-4B-Instruct',
        'description': '4B 参数量，平衡性能与速度，适合大多数场景',
        'min_vram_gb': 8,  # 最小显存需求（8-bit 量化）
        'recommended_vram_gb': 12,  # 推荐显存
    },
    'Qwen3-VL-8B-Instruct': {
        'display_name': 'Qwen3-VL-8B-Instruct',
        'model_path': PROJECT_ROOT / 'models' / 'Qwen3-VL-8B-Instruct',
        'hf_mirror': 'https://hf-mirror.com',
        'hf_model_id': 'Qwen/Qwen3-VL-8B-Instruct',
        'description': '8B 参数量，更强的理解能力，适合复杂场景',
        'min_vram_gb': 12,
        'recommended_vram_gb': 16,
    },
    'Qwen3-VL-30B-A3B-Instruct': {
        'display_name': 'Qwen3-VL-30B-A3B-Instruct',
        'model_path': PROJECT_ROOT / 'models' / 'Qwen3-VL-30B-A3B-Instruct',
        'hf_mirror': 'https://hf-mirror.com',
        'hf_model_id': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'description': '30B 参数量（激活 3B），最强性能，需要高端显卡',
        'min_vram_gb': 16,
        'recommended_vram_gb': 24,
    },
}

# 默认模型
DEFAULT_MODEL = 'Qwen3-VL-4B-Instruct'

# 向后兼容的模型配置（使用默认模型）
MODEL_CONFIG = {
    'model_name': DEFAULT_MODEL,
    'model_path': SUPPORTED_MODELS[DEFAULT_MODEL]['model_path'],
    'hf_mirror': SUPPORTED_MODELS[DEFAULT_MODEL]['hf_mirror'],
    'hf_model_id': SUPPORTED_MODELS[DEFAULT_MODEL]['hf_model_id'],
}

# FFmpeg 配置
FFMPEG_CONFIG = {
    'ffmpeg_dir': PROJECT_ROOT / 'FFmpeg',
    'download_url': 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip',
}

# 视频处理配置
VIDEO_CONFIG = {
    'default_fps': 1.0,  # 默认采样帧率
    'min_fps': 0.5,
    'max_fps': 5.0,
    'default_max_pixels': 360 * 420,  # 默认最大像素数
    'output_suffix': '_no_ads',  # 输出文件后缀
}

# Web UI 配置
UI_CONFIG = {
    'server_name': '0.0.0.0',
    'server_port': 7860,
    'share': False,
    'show_error': True,
    'theme': 'soft',  # Gradio 主题
}

# 日志配置
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# GPU 配置
GPU_CONFIG = {
    'device': 'auto',  # 'auto', 'cuda', 'cpu'
    'use_flash_attn': False,  # 是否默认使用 Flash Attention 2
    'torch_dtype': 'auto',  # 'auto', 'float16', 'bfloat16', 'float32'
}

# 默认提示词
DEFAULT_PROMPTS = {
    'ad_detection': """请仔细分析这个视频，识别其中的广告片段。

请按照以下格式输出：

1. 视频概述：简要描述视频的主要内容
2. 广告检测：
   - 如果发现广告，请列出每个广告片段的大致时间范围（开始时间-结束时间）
   - 描述广告的特征（如品牌名称、产品类型、视觉特征等）
3. 建议：是否建议去除这些片段

广告的常见特征：
- 品牌 logo 或产品展示
- 促销信息、价格信息
- 联系方式（电话、网址等）
- 与主要内容风格明显不同
- 突然的场景切换或音乐变化

请详细分析并给出准确的时间范围。""",
    
    'content_analysis': """请分析这个视频的内容，包括：
1. 主题和内容概述
2. 视觉风格和特点
3. 关键场景和时间点
4. 整体质量评价""",
    
    'scene_detection': """请识别视频中的主要场景，并给出每个场景的时间范围和描述。"""
}

