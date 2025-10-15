"""
工具函数
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import json


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    设置日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )


def format_time(seconds: float) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串 (HH:MM:SS)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"


def format_size(bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        bytes: 字节数
        
    Returns:
        格式化的大小字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} PB"


def parse_time_string(time_str: str) -> Optional[float]:
    """
    解析时间字符串为秒数
    
    Args:
        time_str: 时间字符串，支持格式：
                 - "MM:SS"
                 - "HH:MM:SS"
                 - "XXs" 或 "XX秒"
                 
    Returns:
        秒数，解析失败返回 None
    """
    time_str = time_str.strip()
    
    # 尝试解析 "XXs" 或 "XX秒"
    if time_str.endswith('s') or time_str.endswith('秒'):
        try:
            return float(time_str.rstrip('s秒'))
        except ValueError:
            pass
    
    # 尝试解析 "MM:SS" 或 "HH:MM:SS"
    parts = time_str.split(':')
    try:
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        pass
    
    return None


def merge_overlapping_segments(
    segments: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    合并重叠的时间段
    
    Args:
        segments: 时间段列表 [(start1, end1), (start2, end2), ...]
        
    Returns:
        合并后的时间段列表
    """
    if not segments:
        return []
    
    # 按开始时间排序
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    merged = [sorted_segments[0]]
    
    for current in sorted_segments[1:]:
        last = merged[-1]
        
        # 如果当前段与上一段重叠或相邻
        if current[0] <= last[1]:
            # 合并
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            # 不重叠，添加新段
            merged.append(current)
    
    return merged


def calculate_removed_duration(
    segments: List[Tuple[float, float]]
) -> float:
    """
    计算被移除的总时长
    
    Args:
        segments: 要移除的时间段列表
        
    Returns:
        总时长（秒）
    """
    merged = merge_overlapping_segments(segments)
    return sum(end - start for start, end in merged)


def save_segments_to_json(
    segments: List[Dict],
    output_path: str
):
    """
    保存片段信息到 JSON 文件
    
    Args:
        segments: 片段信息列表
        output_path: 输出文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)


def load_segments_from_json(
    input_path: str
) -> List[Dict]:
    """
    从 JSON 文件加载片段信息
    
    Args:
        input_path: 输入文件路径
        
    Returns:
        片段信息列表
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"文件不存在: {input_path}")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_gpu_availability() -> dict:
    """
    检查 GPU 可用性
    
    Returns:
        GPU 信息字典
    """
    try:
        import torch
        
        info = {
            'cuda_available': torch.cuda.is_available(),
            'device_count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            info['device_count'] = torch.cuda.device_count()
            
            for i in range(info['device_count']):
                device_info = {
                    'id': i,
                    'name': torch.cuda.get_device_name(i),
                    'memory_total': torch.cuda.get_device_properties(i).total_memory,
                    'memory_allocated': torch.cuda.memory_allocated(i),
                    'memory_reserved': torch.cuda.memory_reserved(i),
                }
                info['devices'].append(device_info)
        
        return info
        
    except ImportError:
        return {
            'cuda_available': False,
            'device_count': 0,
            'devices': [],
            'error': 'PyTorch not installed'
        }


def estimate_processing_time(
    video_duration: float,
    fps: float,
    num_segments: int = 1
) -> float:
    """
    估算处理时间
    
    Args:
        video_duration: 视频时长（秒）
        fps: 采样帧率
        num_segments: 片段数量
        
    Returns:
        估算的处理时间（秒）
    """
    # 这是一个简化的估算
    # 实际时间取决于硬件性能、模型大小等因素
    
    # 模型推理时间（假设每帧 0.1 秒）
    inference_time = video_duration * fps * 0.1
    
    # 视频处理时间（假设是视频时长的 10%）
    processing_time = video_duration * 0.1 * num_segments
    
    return inference_time + processing_time


def validate_video_file(file_path: str) -> bool:
    """
    验证视频文件
    
    Args:
        file_path: 视频文件路径
        
    Returns:
        是否有效
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return False
    
    # 检查文件扩展名
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    if file_path.suffix.lower() not in valid_extensions:
        return False
    
    # 检查文件大小
    if file_path.stat().st_size == 0:
        return False
    
    return True


def create_output_filename(
    input_path: str,
    suffix: str = '_no_ads',
    output_dir: Optional[str] = None
) -> str:
    """
    创建输出文件名
    
    Args:
        input_path: 输入文件路径
        suffix: 文件名后缀
        output_dir: 输出目录（可选）
        
    Returns:
        输出文件路径
    """
    input_path = Path(input_path)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_path.parent
    
    output_name = f"{input_path.stem}{suffix}{input_path.suffix}"
    return str(output_dir / output_name)

