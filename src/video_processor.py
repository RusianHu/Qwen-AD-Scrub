"""
视频处理器
负责视频的读取、分析和处理
"""

import subprocess
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self, ffmpeg_path: Optional[str] = None):
        """
        初始化视频处理器
        
        Args:
            ffmpeg_path: FFmpeg 可执行文件路径
        """
        self.project_root = Path(__file__).parent.parent
        
        if ffmpeg_path is None:
            # 自动查找 FFmpeg
            self.ffmpeg_path = self._find_ffmpeg()
        else:
            self.ffmpeg_path = Path(ffmpeg_path)
        
        if not self.ffmpeg_path or not self.ffmpeg_path.exists():
            raise FileNotFoundError(
                f"FFmpeg 未找到\n"
                f"请运行 setup_dependencies.py 安装 FFmpeg"
            )
        
        logger.info(f"使用 FFmpeg: {self.ffmpeg_path}")
    
    def _find_ffmpeg(self) -> Optional[Path]:
        """自动查找 FFmpeg 可执行文件"""
        ffmpeg_dir = self.project_root / "FFmpeg"
        
        if not ffmpeg_dir.exists():
            return None
        
        # 查找 ffmpeg.exe
        for item in ffmpeg_dir.rglob("ffmpeg.exe"):
            if item.exists():
                return item
        
        return None
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        获取视频信息
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            包含视频信息的字典
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        try:
            # 使用 ffprobe 获取视频信息
            ffprobe_path = self.ffmpeg_path.parent / "ffprobe.exe"
            
            cmd = [
                str(ffprobe_path),
                "-v", "quiet",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                str(video_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            info = json.loads(result.stdout)
            
            # 提取关键信息
            video_stream = next(
                (s for s in info['streams'] if s['codec_type'] == 'video'),
                None
            )
            
            if video_stream is None:
                raise ValueError("未找到视频流")
            
            duration = float(info['format'].get('duration', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            fps = eval(video_stream.get('r_frame_rate', '0/1'))
            
            return {
                'duration': duration,
                'width': width,
                'height': height,
                'fps': fps,
                'format': info['format'].get('format_name', ''),
                'size': int(info['format'].get('size', 0)),
                'bitrate': int(info['format'].get('bit_rate', 0))
            }
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            raise
    
    def cut_video(
        self,
        input_path: str,
        output_path: str,
        segments: List[Tuple[float, float]],
        progress_callback=None
    ) -> bool:
        """
        根据时间段切割视频
        
        Args:
            input_path: 输入视频路径
            output_path: 输出视频路径
            segments: 要保留的时间段列表 [(start1, end1), (start2, end2), ...]
            progress_callback: 进度回调函数
            
        Returns:
            是否成功
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"输入视频不存在: {input_path}")
        
        if not segments:
            logger.warning("没有要保留的片段")
            return False
        
        try:
            # 创建输出目录
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果只有一个片段，直接切割
            if len(segments) == 1:
                start, end = segments[0]
                return self._cut_single_segment(
                    input_path, output_path, start, end, progress_callback
                )
            
            # 多个片段需要先分别切割，然后合并
            return self._cut_multiple_segments(
                input_path, output_path, segments, progress_callback
            )
            
        except Exception as e:
            logger.error(f"视频切割失败: {e}")
            return False
    
    def _cut_single_segment(
        self,
        input_path: Path,
        output_path: Path,
        start: float,
        end: float,
        progress_callback=None
    ) -> bool:
        """切割单个片段"""
        duration = end - start

        # 获取原视频信息以保持相同的编码参数
        video_info = self.get_video_info(str(input_path))

        cmd = [
            str(self.ffmpeg_path),
            "-ss", str(start),  # 在输入前指定起始时间（更精确）
            "-i", str(input_path),
            "-t", str(duration),
            "-c:v", "libx264",  # 使用 H.264 编码
            "-preset", "medium",  # 编码速度（medium 平衡质量和速度）
            "-crf", "18",  # 质量参数（18 为高质量，0-51，越小质量越好）
            "-c:a", "aac",  # 音频编码
            "-b:a", "192k",  # 音频比特率
            "-movflags", "+faststart",  # 优化网络播放
            "-y",  # 覆盖输出文件
            str(output_path)
        ]

        logger.info(f"切割片段: {start:.2f}s - {end:.2f}s")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            if progress_callback:
                progress_callback(1.0)

            logger.info(f"✓ 视频已保存: {output_path}")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg 错误: {e.stderr}")
            return False
    
    def _cut_multiple_segments(
        self,
        input_path: Path,
        output_path: Path,
        segments: List[Tuple[float, float]],
        progress_callback=None
    ) -> bool:
        """切割并合并多个片段"""
        temp_dir = output_path.parent / "temp_segments"
        temp_dir.mkdir(exist_ok=True)
        
        temp_files = []
        
        try:
            # 切割每个片段
            for i, (start, end) in enumerate(segments):
                temp_file = temp_dir / f"segment_{i:03d}.mp4"
                
                if self._cut_single_segment(input_path, temp_file, start, end):
                    temp_files.append(temp_file)
                
                if progress_callback:
                    progress_callback((i + 1) / (len(segments) + 1))
            
            if not temp_files:
                logger.error("没有成功切割的片段")
                return False
            
            # 合并片段
            concat_file = temp_dir / "concat_list.txt"
            with open(concat_file, 'w', encoding='utf-8') as f:
                for temp_file in temp_files:
                    # Windows 路径需要转换为正斜杠，并且路径需要转义
                    file_path = str(temp_file.absolute()).replace('\\', '/')
                    f.write(f"file '{file_path}'\n")
            
            cmd = [
                str(self.ffmpeg_path),
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libx264",  # 使用 H.264 编码
                "-preset", "medium",  # 编码速度
                "-crf", "18",  # 质量参数
                "-c:a", "aac",  # 音频编码
                "-b:a", "192k",  # 音频比特率
                "-movflags", "+faststart",  # 优化网络播放
                "-y",
                str(output_path)
            ]
            
            logger.info(f"合并 {len(temp_files)} 个片段...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(f"✓ 视频已保存: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"合并失败: {e}")
            return False
            
        finally:
            # 清理临时文件
            for temp_file in temp_files:
                if temp_file.exists():
                    temp_file.unlink()
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)

