"""
Qwen-AD-Scrub Web UI
基于 Gradio 的视频广告去除工具界面
"""

import gradio as gr
import os
import time
from pathlib import Path
from src.model_loader import ModelLoader
from src.video_processor import VideoProcessor
from src.ad_detector import AdDetector
import logging

# 配置日志：只显示 ERROR 级别
logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 全局变量
model_loader = None
video_processor = None
ad_detector = None

# 输出目录配置
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)  # 确保输出目录存在


def initialize_system():
    """初始化系统"""
    global model_loader, video_processor, ad_detector

    try:
        # 初始化视频处理器
        logger.info("初始化视频处理器...")
        video_processor = VideoProcessor()

        # 初始化模型加载器（不自动加载模型）
        logger.info("初始化模型加载器...")
        model_loader = ModelLoader()

        # 获取可用模型列表
        available_models = model_loader.get_available_models()
        if available_models:
            models_info = "\n".join([f"  - {m}" for m in available_models])
            return f"✓ 系统初始化成功！\n\n发现 {len(available_models)} 个可用模型:\n{models_info}\n\n请选择模型并点击'加载模型'按钮。", gr.update(choices=available_models, value=available_models[0] if available_models else None)
        else:
            return "✓ 系统初始化成功，但未找到可用模型。\n请运行 setup_dependencies.py 下载模型。", gr.update(choices=[], value=None)
    except Exception as e:
        return f"✗ 系统初始化失败: {str(e)}", gr.update(choices=[], value=None)


def get_available_models():
    """获取可用模型列表"""
    global model_loader

    if model_loader is None:
        return []

    return model_loader.get_available_models()


def load_model_ui(model_name, use_flash_attn, use_8bit):
    """加载模型 UI 回调"""
    global model_loader, ad_detector

    if model_loader is None:
        return "请先初始化系统"

    if not model_name:
        return "请选择要加载的模型"

    try:
        # 显示加载信息
        info_msg = f"正在加载模型: {model_name}\n请稍候...\n\n"
        if use_8bit:
            info_msg += "✓ 已启用 8-bit 量化（节省约 50% 显存）\n"
        if use_flash_attn:
            info_msg += "✓ 已启用 Flash Attention 2 加速\n"
        yield info_msg

        # 加载模型（会自动处理模型切换）
        model_loader.load_model(
            model_name=model_name,
            use_flash_attn=use_flash_attn,
            use_8bit=use_8bit
        )

        # 重新创建检测器
        ad_detector = AdDetector(model_loader)

        yield f"✓ 模型 {model_name} 加载完成！可以开始使用了。"
    except Exception as e:
        yield f"✗ 模型加载失败: {str(e)}"


def switch_model_ui(new_model_name, use_flash_attn, use_8bit, progress=gr.Progress()):
    """切换模型 UI 回调"""
    # 在函数的绝对开始就初始化进度条
    progress(0, desc="初始化...")

    global model_loader, ad_detector

    if model_loader is None:
        progress(0, desc="请先初始化系统")
        return "请先初始化系统"

    if not new_model_name:
        progress(0, desc="请选择要切换的模型")
        return "请选择要切换的模型"

    progress(0.05, desc="准备切换模型...")

    try:
        current_model = model_loader.current_model_name

        if current_model == new_model_name and model_loader.is_loaded():
            return f"模型 {new_model_name} 已经加载，无需切换"

        progress(0.1, desc=f"正在切换到模型: {new_model_name}...")

        # 显示切换信息
        info_msg = f"正在切换模型:\n"
        info_msg += f"  当前: {current_model or '无'}\n"
        info_msg += f"  目标: {new_model_name}\n\n"

        if current_model:
            progress(0.2, desc="卸载当前模型...")
            info_msg += "正在卸载当前模型...\n"

        yield info_msg

        progress(0.4, desc="加载新模型...")

        # 切换模型
        model_loader.switch_model(
            new_model_name=new_model_name,
            use_flash_attn=use_flash_attn,
            use_8bit=use_8bit
        )

        progress(0.8, desc="重新初始化检测器...")

        # 重新创建检测器
        ad_detector = AdDetector(model_loader)

        progress(1.0, desc="切换完成!")

        yield f"✓ 成功切换到模型: {new_model_name}\n可以开始使用了。"

    except Exception as e:
        yield f"✗ 模型切换失败: {str(e)}"


def analyze_video_ui(video_file, fps, custom_prompt, progress=gr.Progress()):
    """分析视频 UI 回调"""
    # 在函数的绝对开始就初始化进度条
    progress(0, desc="初始化...")

    global ad_detector

    if ad_detector is None:
        progress(0, desc="请先加载模型")
        return "请先加载模型", None, "", gr.update(interactive=True)

    if video_file is None:
        progress(0, desc="请上传视频文件")
        return "请上传视频文件", None, "", gr.update(interactive=True)

    progress(0.05, desc="准备分析...")

    try:
        # 记录视频文件路径
        logger.info(f"收到视频文件: {video_file}")
        logger.info(f"文件类型: {type(video_file)}")

        # 确保路径存在
        video_path = Path(video_file)
        if not video_path.exists():
            progress(0, desc="视频文件不存在")
            return f"视频文件不存在: {video_file}", None, "", gr.update(interactive=True)

        logger.info(f"视频文件大小: {video_path.stat().st_size / (1024*1024):.2f} MB")

        # 进度回调
        def progress_callback(value, desc="处理中..."):
            progress(value, desc=desc)

        # 检测广告
        progress(0.1, desc="开始分析视频...")

        result = ad_detector.detect_ads_in_video(
            video_path=str(video_path),  # 确保传递字符串路径
            fps=fps,
            custom_prompt=custom_prompt if custom_prompt.strip() else None,
            progress_callback=progress_callback
        )

        # 格式化输出
        output_text = f"""
## 分析结果

### 📊 检测状态
{'🔴 发现广告片段' if result['has_ads'] else '🟢 未发现明显广告'}

### 📝 模型输出
{result['raw_output']}

### ⏱️ 检测到的广告片段
"""

        if result['ad_segments']:
            for i, seg in enumerate(result['ad_segments'], 1):
                output_text += f"\n**片段 {i}:** {seg['start']}s - {seg['end']}s (时长: {seg['duration']}s)"
        else:
            output_text += "\n未检测到明确的时间范围"

        # 返回结果和片段信息，并重新启用按钮
        segments_info = result['ad_segments'] if result['ad_segments'] else None

        return output_text, segments_info, result['raw_output'], gr.update(interactive=True)

    except Exception as e:
        logger.error(f"分析失败: {e}")
        return f"✗ 分析失败: {str(e)}", None, "", gr.update(interactive=True)


def process_video_ui(video_file, segments_json, progress=gr.Progress()):
    """处理视频 UI 回调"""
    # 在函数的绝对开始就初始化进度条
    progress(0, desc="初始化...")

    global video_processor

    if video_processor is None:
        progress(0, desc="视频处理器未初始化")
        return "视频处理器未初始化", None

    if video_file is None:
        progress(0, desc="请上传视频文件")
        return "请上传视频文件", None

    if not segments_json:
        progress(0, desc="请先分析视频")
        return "请先分析视频以获取广告片段信息", None

    progress(0.05, desc="准备处理...")

    try:
        # 解析片段信息
        import json
        if isinstance(segments_json, str):
            segments = json.loads(segments_json)
        else:
            segments = segments_json
        
        if not segments:
            return "没有需要去除的广告片段", None
        
        # 获取视频信息
        progress(0.1, desc="获取视频信息...")
        video_info = video_processor.get_video_info(video_file)
        duration = video_info['duration']
        
        # 计算要保留的片段（去除广告后的片段）
        progress(0.2, desc="计算保留片段...")
        keep_segments = []
        last_end = 0.0
        
        for seg in segments:
            start = float(seg['start'])
            end = float(seg['end'])
            
            # 添加广告前的片段
            if start > last_end:
                keep_segments.append((last_end, start))
            
            last_end = end
        
        # 添加最后一个片段
        if last_end < duration:
            keep_segments.append((last_end, duration))
        
        if not keep_segments:
            return "去除广告后没有剩余内容", None
        
        # 生成输出文件名（保存到 output 目录）
        input_path = Path(video_file)
        output_path = OUTPUT_DIR / f"{input_path.stem}_no_ads{input_path.suffix}"
        
        # 处理视频
        progress(0.3, desc="正在处理视频...")
        
        def ffmpeg_progress(value):
            progress(0.3 + value * 0.6, desc="正在处理视频...")
        
        success = video_processor.cut_video(
            input_path=video_file,
            output_path=str(output_path),
            segments=keep_segments,
            progress_callback=ffmpeg_progress
        )
        
        if success:
            progress(1.0, desc="处理完成!")
            return f"✓ 视频处理完成！\n保存位置: {output_path}", str(output_path)
        else:
            return "✗ 视频处理失败", None
            
    except Exception as e:
        logger.error(f"处理失败: {e}")
        return f"✗ 处理失败: {str(e)}", None


def get_video_info_ui(video_file):
    """获取视频信息 UI 回调"""
    global video_processor
    
    if video_processor is None:
        return "视频处理器未初始化"
    
    if video_file is None:
        return "请上传视频文件"
    
    try:
        info = video_processor.get_video_info(video_file)
        
        output = f"""
## 📹 视频信息

- **时长:** {info['duration']:.2f} 秒 ({info['duration']/60:.2f} 分钟)
- **分辨率:** {info['width']} x {info['height']}
- **帧率:** {info['fps']:.2f} fps
- **格式:** {info['format']}
- **文件大小:** {info['size'] / 1024 / 1024:.2f} MB
- **比特率:** {info['bitrate'] / 1000:.0f} kbps
"""
        return output
    except Exception as e:
        return f"✗ 获取信息失败: {str(e)}"


# 创建 Gradio 界面
def create_ui():
    """创建 Gradio UI"""
    
    # 自定义 CSS - bilibili 风格 + Material Design
    custom_css = """
    .gradio-container {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #00a1d6;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Qwen-AD-Scrub", theme=gr.themes.Soft()) as app:

        # 标题
        gr.HTML("""
        <div class="main-title">
            <h1>🎬 Qwen-AD-Scrub</h1>
            <p>基于 Qwen3-VL-4B-Instruct 的智能视频广告去除工具</p>
        </div>
        """)

        # 状态变量
        segments_state = gr.State(None)

        # 系统初始化和模型加载模块（上方，默认展开）
        with gr.Accordion("⚙️ 系统初始化与模型管理", open=True):
            gr.Markdown("""
            <div class="info-box">
            <strong>💡 使用提示:</strong>
            <ul>
                <li>首次使用需要先<strong>初始化系统</strong>，然后<strong>选择并加载模型</strong></li>
                <li>支持多个不同参数量的 Qwen3-VL 模型（4B、8B、30B 等）</li>
                <li>可以在运行时<strong>动态切换模型</strong>，系统会自动释放旧模型显存</li>
                <li><strong>8-bit 量化</strong>: 默认启用，可节省约 50% 显存，对精度影响很小</li>
                <li>Flash Attention 2 需要支持的 GPU (Ampere 架构及以上，如 RTX 30/40/50 系列)</li>
            </ul>
            </div>
            """)

            # 步骤 1: 系统初始化
            gr.Markdown("### 📋 步骤 1: 初始化系统")
            with gr.Row():
                init_btn = gr.Button("🚀 初始化系统", variant="primary", size="lg", scale=1)
                init_output = gr.Textbox(label="初始化状态", lines=3, interactive=False, scale=3)

            gr.Markdown("---")  # 分隔线

            # 步骤 2: 模型选择与加载
            gr.Markdown("### 🤖 步骤 2: 选择并加载模型")

            model_selector = gr.Dropdown(
                label="选择模型",
                choices=[],
                value=None,
                info="从 models 文件夹中扫描到的可用模型"
            )

            with gr.Row():
                use_8bit = gr.Checkbox(
                    label="8-bit 量化",
                    value=True,
                    info="推荐开启，节省约 50% 显存"
                )
                use_flash_attn = gr.Checkbox(
                    label="Flash Attention 2",
                    value=False,
                    info="需要单独安装（Windows 上较复杂）"
                )

            with gr.Row():
                load_model_btn = gr.Button("📦 加载模型", variant="primary", size="lg", scale=1)
                switch_model_btn = gr.Button("🔄 切换模型", variant="secondary", size="lg", scale=1)

            model_output = gr.Textbox(label="模型状态", lines=3, interactive=False)

        # 主功能标签页
        with gr.Tabs():
            # Tab 1: 视频分析
            with gr.Tab("🔍 视频分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="上传视频")

                        gr.Markdown("### 分析参数")
                        fps_slider = gr.Slider(
                            minimum=0.2, maximum=2.0, value=0.3, step=0.1,
                            label="采样帧率 (fps)",
                            info="每秒提取多少帧进行分析，越高越精确但显存占用越大。长视频建议 0.2-0.3"
                        )

                        custom_prompt_input = gr.Textbox(
                            label="自定义提示词 (可选)",
                            placeholder="留空使用默认提示词...",
                            lines=5
                        )

                        analyze_btn = gr.Button("🔍 开始分析", variant="primary", size="lg")
                        info_btn = gr.Button("📊 获取视频信息", size="sm")

                    with gr.Column(scale=1):
                        analysis_output = gr.Markdown(label="分析结果")
                        video_info_output = gr.Markdown(label="视频信息")

            # Tab 2: 视频处理
            with gr.Tab("✂️ 视频处理"):
                gr.Markdown("### 去除广告")
                gr.Markdown("分析完成后，点击下方按钮去除检测到的广告片段")
                
                process_btn = gr.Button("✂️ 去除广告并导出", variant="primary", size="lg")
                process_output = gr.Textbox(label="处理状态", lines=3)
                output_video = gr.File(label="处理后的视频")
                
                gr.Markdown("""
                <div class="info-box">
                <strong>⚠️ 注意:</strong>
                <ul>
                    <li>处理大文件可能需要较长时间</li>
                    <li>确保有足够的磁盘空间</li>
                    <li>处理后的视频将保存在项目的 <code>output</code> 文件夹中</li>
                </ul>
                </div>
                """)
        
        # 事件绑定
        init_btn.click(
            fn=initialize_system,
            outputs=[init_output, model_selector]
        )

        load_model_btn.click(
            fn=load_model_ui,
            inputs=[model_selector, use_flash_attn, use_8bit],
            outputs=model_output
        )

        switch_model_btn.click(
            fn=switch_model_ui,
            inputs=[model_selector, use_flash_attn, use_8bit],
            outputs=model_output,
            show_progress="full"  # 显示完整的进度条和遮罩
        )

        # 分析按钮点击事件：先禁用按钮，然后执行分析，最后重新启用
        analyze_btn.click(
            fn=lambda: gr.update(interactive=False),
            inputs=None,
            outputs=analyze_btn,
            queue=False  # 立即执行，不排队
        ).then(
            fn=analyze_video_ui,
            inputs=[video_input, fps_slider, custom_prompt_input],
            outputs=[analysis_output, segments_state, gr.Textbox(visible=False), analyze_btn],
            show_progress="full",  # 显示完整的进度条和遮罩
            show_api=False  # 隐藏 API 信息
        )

        info_btn.click(
            fn=get_video_info_ui,
            inputs=video_input,
            outputs=video_info_output
        )

        process_btn.click(
            fn=process_video_ui,
            inputs=[video_input, segments_state],
            outputs=[process_output, output_video],
            show_progress="full"  # 显示完整的进度条和遮罩
        )

    return app


def check_dependencies():
    """检查关键依赖是否安装（静默检查）"""
    missing_deps = []

    # 检查 bitsandbytes（8-bit 量化需要）
    try:
        import bitsandbytes
    except ImportError:
        missing_deps.append("bitsandbytes")

    # 检查 transformers
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")

    # 检查 qwen-vl-utils
    try:
        import qwen_vl_utils
    except ImportError:
        missing_deps.append("qwen-vl-utils")

    if missing_deps:
        logger.error(f"缺少依赖: {', '.join(missing_deps)}")
        logger.error(f"安装命令: pip install {' '.join(missing_deps)} -i https://pypi.tuna.tsinghua.edu.cn/simple")

        # 如果缺少必需依赖，返回 False
        if "transformers" in missing_deps or "qwen-vl-utils" in missing_deps:
            return False

    return True


if __name__ == "__main__":
    # 静默检查依赖
    if not check_dependencies():
        import sys
        sys.exit(1)

    # 创建并启动应用
    try:
        app = create_ui()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"启动失败: {e}")
        import sys
        sys.exit(1)

