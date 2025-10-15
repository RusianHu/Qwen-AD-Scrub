"""
广告检测器
使用 Qwen3-VL 模型检测视频中的广告片段
"""

import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from qwen_vl_utils import process_vision_info
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdDetector:
    """广告检测器"""
    
    def __init__(self, model_loader):
        """
        初始化广告检测器
        
        Args:
            model_loader: ModelLoader 实例
        """
        self.model_loader = model_loader
        
        if not self.model_loader.is_loaded():
            raise RuntimeError("模型未加载，请先加载模型")
        
        self.model = self.model_loader.get_model()
        self.processor = self.model_loader.get_processor()
    
    def detect_ads_in_video(
        self,
        video_path: str,
        fps: float = 0.3,  # 降低默认采样率，减少显存占用（从 0.5 降到 0.3）
        min_pixels: int = 4 * 28 * 28,  # 最小像素数 (3136)，用于低分辨率视频
        max_pixels: int = 64 * 28 * 28,  # 最大像素数 (50176)，极限节省显存
        custom_prompt: Optional[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        检测视频中的广告

        Args:
            video_path: 视频文件路径
            fps: 采样帧率（每秒提取多少帧）
            min_pixels: 最小像素数（必须小于等于 max_pixels）
            max_pixels: 最大像素数（用于控制显存占用）
            custom_prompt: 自定义提示词
            progress_callback: 进度回调函数

        Returns:
            检测结果字典，包含广告片段信息

        Note:
            像素参数说明：
            - min_pixels 和 max_pixels 必须满足: min_pixels <= max_pixels
            - 推荐设置（根据显存和视频长度）：
              * 极限节省 (长视频 > 3分钟): min_pixels=4*28*28, max_pixels=64*28*28, fps=0.2-0.3
              * 低显存 (< 8GB): min_pixels=4*28*28, max_pixels=128*28*28, fps=0.3-0.5
              * 中等显存 (8-16GB): min_pixels=16*28*28, max_pixels=256*28*28, fps=0.5-1.0
              * 高显存 (> 16GB): min_pixels=256*28*28, max_pixels=512*28*28, fps=1.0-2.0
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        logger.info(f"开始分析视频: {video_path.name}")
        logger.info(f"采样帧率: {fps} fps")
        
        # 构建提示词
        if custom_prompt is None:
            prompt = self._build_default_prompt()
        else:
            prompt = custom_prompt

        # 处理视频路径
        # 注意：直接使用本地文件路径，不使用 file:// 协议
        # 因为 decord 和 torchvision 对 file:// 协议的支持有问题，
        # 特别是当路径包含中文字符时
        video_path_str = str(video_path.absolute())

        logger.info(f"视频路径: {video_path_str}")

        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path_str,
                        "min_pixels": min_pixels,
                        "max_pixels": max_pixels,
                        "fps": fps,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        try:
            # 处理输入
            if progress_callback:
                progress_callback(0.3, "正在处理视频...")

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            logger.info("处理视频输入...")

            # 尝试处理视频输入
            try:
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                logger.info("✓ process_vision_info 成功（带 video_kwargs）")
            except Exception as e:
                logger.warning(f"process_vision_info 失败: {e}")
                logger.info("尝试使用备用方法...")

                # 备用方案：不使用 return_video_kwargs
                try:
                    image_inputs, video_inputs = process_vision_info(messages)
                    video_kwargs = None  # 先设为 None
                    logger.info("✓ process_vision_info 成功（不带 video_kwargs）")
                except Exception as e2:
                    logger.error(f"备用方法也失败: {e2}")
                    raise RuntimeError(f"无法处理视频输入: {e2}") from e2

            # 检查视频是否成功加载
            if video_inputs is None:
                raise RuntimeError(
                    "视频加载失败，video_inputs 为 None。\n"
                    "可能的原因：\n"
                    "1. 视频文件格式不支持\n"
                    "2. 视频文件损坏\n"
                    "3. 路径包含特殊字符\n"
                    f"视频路径: {video_path_str}"
                )

            logger.info(f"✓ 视频加载成功")
            logger.info(f"  video_inputs type: {type(video_inputs)}")
            logger.info(f"  video_inputs shape: {video_inputs.shape if hasattr(video_inputs, 'shape') else 'N/A'}")

            # 安全处理 video_kwargs
            # 只有当 video_inputs 不为 None 时才使用 video_kwargs
            if video_kwargs is None:
                video_kwargs = {}

            # 清理 video_kwargs，移除不需要的键
            # processor 不接受 video_fps 参数（会被忽略）
            # 同时需要处理 fps 参数的类型问题
            cleaned_kwargs = {}
            for key, value in video_kwargs.items():
                if key == 'video_fps':
                    # 跳过 video_fps，processor 不需要这个参数
                    continue
                elif key == 'fps':
                    # 如果 fps 是列表，取第一个元素
                    if isinstance(value, list) and len(value) > 0:
                        cleaned_kwargs[key] = value[0]
                    elif isinstance(value, (int, float)):
                        cleaned_kwargs[key] = value
                    # 否则跳过
                else:
                    cleaned_kwargs[key] = value

            logger.info(f"  原始 video_kwargs: {video_kwargs}")
            logger.info(f"  清理后 video_kwargs: {cleaned_kwargs}")

            # 构建 processor 输入
            processor_kwargs = {
                'text': [text],
                'images': image_inputs,
                'videos': video_inputs,
                'padding': True,
                'return_tensors': 'pt',
            }

            # 只有当有视频输入时才添加清理后的 video_kwargs
            if video_inputs is not None and cleaned_kwargs:
                processor_kwargs.update(cleaned_kwargs)

            logger.info("调用 processor...")
            inputs = self.processor(**processor_kwargs)
            
            # 清理 GPU 缓存，为推理腾出空间
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("已清理 GPU 缓存")

            # 移动到设备（如果使用 8-bit 量化，模型已经在正确的设备上）
            # 只需要移动输入
            if not hasattr(self.model, 'is_loaded_in_8bit') or not self.model.is_loaded_in_8bit:
                device = next(self.model.parameters()).device
                inputs = inputs.to(device)

            # 推理
            if progress_callback:
                progress_callback(0.6, "正在分析内容...")

            logger.info("开始推理...")

            # 显示推理前的显存使用情况
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"推理前 GPU {i}: 已分配 {allocated:.2f}GB / 保留 {reserved:.2f}GB")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False
                )

            # 推理后清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("推理完成，已清理 GPU 缓存")
            
            # 解码输出
            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            if progress_callback:
                progress_callback(0.9, "正在解析结果...")
            
            logger.info(f"模型输出:\n{output_text}")
            
            # 解析输出
            result = self._parse_detection_result(output_text)
            
            if progress_callback:
                progress_callback(1.0, "分析完成!")
            
            return result

        except Exception as e:
            import traceback
            logger.error(f"检测失败: {e}")
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
            logger.error(f"视频路径: {video_path_str}")
            logger.error(f"messages: {messages}")
            raise RuntimeError(f"视频分析失败: {str(e)}") from e
    
    def _build_default_prompt(self) -> str:
        """构建默认的检测提示词"""
        prompt = """请仔细分析这个视频，识别其中的广告片段。

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

请详细分析并给出准确的时间范围。"""
        
        return prompt
    
    def _parse_detection_result(self, output_text: str) -> Dict:
        """
        解析模型输出结果
        
        Args:
            output_text: 模型输出文本
            
        Returns:
            解析后的结果字典
        """
        result = {
            'raw_output': output_text,
            'has_ads': False,
            'ad_segments': [],
            'summary': '',
            'recommendations': ''
        }
        
        # 简单的文本解析
        # 这里可以根据实际输出格式进行更复杂的解析
        
        # 检查是否提到广告
        ad_keywords = ['广告', '推广', '促销', 'logo', '品牌', '产品']
        if any(keyword in output_text for keyword in ad_keywords):
            result['has_ads'] = True
        
        # 尝试提取时间范围
        # 这是一个简化的实现，实际可能需要更复杂的解析逻辑
        import re

        # 匹配类似 "0:30-1:00" 或 "30秒-60秒" 或 "0:02:25 - 0:02:30" 的时间范围
        time_patterns = [
            r'(\d+):(\d+):(\d+)\s*-\s*(\d+):(\d+):(\d+)',  # H:MM:SS-H:MM:SS
            r'(\d+):(\d+)\s*-\s*(\d+):(\d+)',  # MM:SS-MM:SS
            r'(\d+)秒\s*-\s*(\d+)秒',  # XX秒-YY秒
            r'(\d+)\s*-\s*(\d+)\s*秒',  # XX-YY秒
        ]

        for pattern in time_patterns:
            matches = re.findall(pattern, output_text)
            for match in matches:
                try:
                    if len(match) == 6:  # H:MM:SS format
                        start = int(match[0]) * 3600 + int(match[1]) * 60 + int(match[2])
                        end = int(match[3]) * 3600 + int(match[4]) * 60 + int(match[5])
                    elif len(match) == 4:  # MM:SS format
                        start = int(match[0]) * 60 + int(match[1])
                        end = int(match[2]) * 60 + int(match[3])
                    elif len(match) == 2:  # 秒 format
                        start = int(match[0])
                        end = int(match[1])
                    else:
                        continue

                    if start < end:
                        result['ad_segments'].append({
                            'start': start,
                            'end': end,
                            'duration': end - start
                        })
                        logger.info(f"✓ 解析到广告片段: {start}s - {end}s (时长: {end - start}s)")
                except (ValueError, IndexError) as e:
                    logger.warning(f"解析时间失败: {match}, 错误: {e}")
                    continue
        
        # 提取摘要
        lines = output_text.split('\n')
        for i, line in enumerate(lines):
            if '视频概述' in line or '概述' in line:
                if i + 1 < len(lines):
                    result['summary'] = lines[i + 1].strip()
            elif '建议' in line:
                if i + 1 < len(lines):
                    result['recommendations'] = lines[i + 1].strip()
        
        return result
    
    def analyze_with_custom_prompt(
        self,
        video_path: str,
        prompt: str,
        fps: float = 0.3,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 64 * 28 * 28
    ) -> str:
        """
        使用自定义提示词分析视频

        Args:
            video_path: 视频文件路径
            prompt: 自定义提示词
            fps: 采样帧率
            min_pixels: 最小像素数
            max_pixels: 最大像素数

        Returns:
            模型输出文本
        """
        result = self.detect_ads_in_video(
            video_path=video_path,
            fps=fps,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            custom_prompt=prompt
        )

        return result['raw_output']

