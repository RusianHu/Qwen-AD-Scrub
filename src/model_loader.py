"""
模型加载器
负责加载和管理 Qwen3-VL 系列模型
支持多模型动态切换
"""

import torch
from pathlib import Path
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from typing import Optional, List, Dict
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelLoader:
    """Qwen3-VL 模型加载器，支持多模型管理"""

    def __init__(self, model_name: Optional[str] = None, device: str = "auto"):
        """
        初始化模型加载器

        Args:
            model_name: 模型名称（如 'Qwen3-VL-4B-Instruct'），默认为 None（不自动加载）
            device: 设备类型，"auto" 自动选择，"cuda" GPU，"cpu" CPU
        """
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "models"
        self.device = device

        # 当前加载的模型
        self.current_model_name = None
        self.model = None
        self.processor = None

        # 记录当前加载配置，用于检测配置变化
        self.current_config = {
            'use_flash_attn': None,
            'use_8bit': None,
            'use_gradient_checkpointing': None
        }

        # 扫描可用模型
        self.available_models = self._scan_available_models()

        # 如果指定了模型名称，验证其存在性
        if model_name is not None:
            if model_name not in self.available_models:
                raise ValueError(
                    f"模型 '{model_name}' 不可用。\n"
                    f"可用模型: {list(self.available_models.keys())}\n"
                    f"请运行 setup_dependencies.py 下载模型"
                )
            self.current_model_name = model_name

    def _scan_available_models(self) -> Dict[str, Path]:
        """
        扫描 models 文件夹下的可用模型

        Returns:
            字典，键为模型名称，值为模型路径
        """
        available = {}

        if not self.models_dir.exists():
            logger.warning(f"模型目录不存在: {self.models_dir}")
            return available

        # 扫描所有子目录
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # 检查是否包含必要的模型文件
            config_file = model_dir / "config.json"
            if config_file.exists():
                try:
                    # 验证是否为 Qwen3-VL 模型
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        # 检查模型架构
                        if 'model_type' in config and 'qwen' in config['model_type'].lower():
                            available[model_dir.name] = model_dir
                            logger.info(f"发现可用模型: {model_dir.name}")
                except Exception as e:
                    logger.warning(f"读取模型配置失败 {model_dir.name}: {e}")

        if not available:
            logger.warning("未找到任何可用模型，请运行 setup_dependencies.py 下载模型")

        return available

    def get_available_models(self) -> List[str]:
        """
        获取可用模型列表

        Returns:
            模型名称列表
        """
        return list(self.available_models.keys())

    def get_model_info(self, model_name: Optional[str] = None) -> Dict:
        """
        获取模型信息

        Args:
            model_name: 模型名称，默认为当前加载的模型

        Returns:
            模型信息字典
        """
        if model_name is None:
            model_name = self.current_model_name

        if model_name is None:
            raise ValueError("未指定模型名称且没有加载任何模型")

        if model_name not in self.available_models:
            raise ValueError(f"模型 '{model_name}' 不可用")

        model_path = self.available_models[model_name]
        config_file = model_path / "config.json"

        info = {
            'name': model_name,
            'path': str(model_path),
            'is_loaded': model_name == self.current_model_name and self.is_loaded(),
        }

        # 读取配置文件获取更多信息
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                info['model_type'] = config.get('model_type', 'unknown')
                info['hidden_size'] = config.get('hidden_size', 'unknown')
                info['num_hidden_layers'] = config.get('num_hidden_layers', 'unknown')
        except Exception as e:
            logger.warning(f"读取模型配置失败: {e}")

        return info
    
    def load_model(
        self,
        model_name: Optional[str] = None,
        use_flash_attn: bool = False,
        use_8bit: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        """
        加载模型和处理器

        Args:
            model_name: 要加载的模型名称，默认使用初始化时指定的模型
            use_flash_attn: 是否使用 Flash Attention 2 加速
            use_8bit: 是否使用 8-bit 量化（推荐，可节省约 50% 显存）
            use_gradient_checkpointing: 是否使用梯度检查点（推荐，可节省显存，推理时无影响）
        """
        # 确定要加载的模型
        if model_name is None:
            if self.current_model_name is None:
                raise ValueError("未指定模型名称，请在初始化时指定或在调用 load_model 时传入")
            model_name = self.current_model_name

        # 验证模型是否可用
        if model_name not in self.available_models:
            raise ValueError(
                f"模型 '{model_name}' 不可用。\n"
                f"可用模型: {list(self.available_models.keys())}\n"
                f"请运行 setup_dependencies.py 下载模型"
            )

        # 检查是否需要重新加载
        new_config = {
            'use_flash_attn': use_flash_attn,
            'use_8bit': use_8bit,
            'use_gradient_checkpointing': use_gradient_checkpointing
        }

        config_changed = (
            self.is_loaded() and
            model_name == self.current_model_name and
            new_config != self.current_config
        )

        if config_changed:
            logger.info(f"检测到配置变化，需要重新加载模型 {model_name}")
            logger.info(f"  旧配置: {self.current_config}")
            logger.info(f"  新配置: {new_config}")
            self.unload_model()

        # 如果要加载的模型与当前模型不同，先卸载当前模型
        elif self.is_loaded() and model_name != self.current_model_name:
            logger.info(f"检测到模型切换: {self.current_model_name} -> {model_name}")
            logger.info("正在卸载当前模型...")
            self.unload_model()

        # 如果模型已加载且配置相同，跳过
        elif self.is_loaded() and model_name == self.current_model_name:
            logger.info(f"模型 {model_name} 已加载且配置相同，跳过加载")
            return

        model_path = self.available_models[model_name]
        logger.info(f"正在加载模型: {model_name}")
        logger.info(f"模型路径: {model_path}")

        try:
            # 加载处理器
            logger.info("加载处理器...")
            self.processor = AutoProcessor.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )

            # 加载模型
            logger.info("加载模型...")
            model_kwargs = {
                "trust_remote_code": True
            }

            # 配置量化
            if use_8bit:
                logger.info("启用 8-bit 量化（节省显存）")

                # 创建 BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )

                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"  # 自动分配到多卡
            else:
                model_kwargs["torch_dtype"] = "auto"
                model_kwargs["device_map"] = self.device

            # 如果使用 Flash Attention 2
            if use_flash_attn:
                # 检查 flash_attn 是否已安装
                try:
                    import flash_attn
                    logger.info("启用 Flash Attention 2")
                    model_kwargs["attn_implementation"] = "flash_attention_2"
                    if not use_8bit:
                        model_kwargs["torch_dtype"] = torch.bfloat16
                except ImportError:
                    logger.warning(
                        "⚠️ Flash Attention 2 未安装，将使用标准 Attention\n"
                        "如需安装，请运行: pip install flash-attn --no-build-isolation\n"
                        "注意：Windows 上安装 flash-attn 可能需要 Visual Studio 和 CUDA 工具链"
                    )
                    use_flash_attn = False

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                str(model_path),
                **model_kwargs
            )

            # 启用梯度检查点（仅在推理时也能节省显存）
            if use_gradient_checkpointing:
                logger.info("启用梯度检查点（节省显存）")
                self.model.gradient_checkpointing_enable()

            # 更新当前模型名称和配置
            self.current_model_name = model_name
            self.current_config = new_config

            logger.info(f"✓ 模型 {model_name} 加载完成!")

            # 显示设备信息
            if torch.cuda.is_available():
                logger.info(f"使用 GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"GPU 数量: {torch.cuda.device_count()}")

                # 显示每个 GPU 的显存使用情况
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i} ({torch.cuda.get_device_name(i)}): "
                              f"已分配 {allocated:.2f}GB / 保留 {reserved:.2f}GB / 总计 {total:.2f}GB")
            else:
                logger.info("使用 CPU")

            return True

        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            # 加载失败时清理状态
            self.model = None
            self.processor = None
            self.current_model_name = None
            self.current_config = {
                'use_flash_attn': None,
                'use_8bit': None,
                'use_gradient_checkpointing': None
            }
            raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.model is not None and self.processor is not None
    
    def get_model(self):
        """获取模型实例"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")
        return self.model
    
    def get_processor(self):
        """获取处理器实例"""
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")
        return self.processor
    
    def unload_model(self):
        """
        安全卸载模型释放显存

        确保：
        1. 正确释放模型和处理器对象
        2. 清理 CUDA 缓存
        3. 记录显存释放情况
        """
        if not self.is_loaded():
            logger.info("没有已加载的模型需要卸载")
            return

        model_name = self.current_model_name
        logger.info(f"正在卸载模型: {model_name}")

        # 记录卸载前的显存使用
        if torch.cuda.is_available():
            before_allocated = sum(
                torch.cuda.memory_allocated(i) / 1024**3
                for i in range(torch.cuda.device_count())
            )
            logger.info(f"卸载前显存占用: {before_allocated:.2f}GB")

        # 删除模型和处理器
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 多次清理确保彻底释放
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            # 记录卸载后的显存使用
            after_allocated = sum(
                torch.cuda.memory_allocated(i) / 1024**3
                for i in range(torch.cuda.device_count())
            )
            freed = before_allocated - after_allocated
            logger.info(f"卸载后显存占用: {after_allocated:.2f}GB")
            logger.info(f"释放显存: {freed:.2f}GB")

        self.current_model_name = None
        # 清理配置
        self.current_config = {
            'use_flash_attn': None,
            'use_8bit': None,
            'use_gradient_checkpointing': None
        }
        logger.info(f"✓ 模型 {model_name} 已成功卸载")

    def switch_model(
        self,
        new_model_name: str,
        use_flash_attn: bool = False,
        use_8bit: bool = True,
        use_gradient_checkpointing: bool = True
    ):
        """
        切换到新模型

        这是一个便捷方法，会自动处理卸载和加载过程

        Args:
            new_model_name: 新模型名称
            use_flash_attn: 是否使用 Flash Attention 2
            use_8bit: 是否使用 8-bit 量化
            use_gradient_checkpointing: 是否使用梯度检查点

        Returns:
            是否切换成功
        """
        if new_model_name not in self.available_models:
            raise ValueError(
                f"模型 '{new_model_name}' 不可用。\n"
                f"可用模型: {list(self.available_models.keys())}"
            )

        # 检查是否需要切换（考虑配置变化）
        new_config = {
            'use_flash_attn': use_flash_attn,
            'use_8bit': use_8bit,
            'use_gradient_checkpointing': use_gradient_checkpointing
        }

        if (self.current_model_name == new_model_name and
            self.is_loaded() and
            new_config == self.current_config):
            logger.info(f"模型 {new_model_name} 已加载且配置相同，无需切换")
            return True

        logger.info(f"开始切换模型: {self.current_model_name or '无'} -> {new_model_name}")

        # 备份当前状态，以便失败时恢复
        backup_model_name = self.current_model_name
        backup_config = self.current_config.copy()
        backup_model = self.model
        backup_processor = self.processor

        try:
            # load_model 方法会自动处理卸载和加载
            self.load_model(
                model_name=new_model_name,
                use_flash_attn=use_flash_attn,
                use_8bit=use_8bit,
                use_gradient_checkpointing=use_gradient_checkpointing
            )
            logger.info(f"✓ 成功切换到模型: {new_model_name}")
            return True
        except Exception as e:
            logger.error(f"模型切换失败: {e}")

            # 尝试恢复到之前的状态
            if backup_model is not None and backup_processor is not None:
                logger.warning(f"尝试恢复到之前的模型: {backup_model_name}")
                try:
                    self.model = backup_model
                    self.processor = backup_processor
                    self.current_model_name = backup_model_name
                    self.current_config = backup_config
                    logger.info(f"✓ 已恢复到模型: {backup_model_name}")
                except Exception as restore_error:
                    logger.error(f"恢复失败: {restore_error}")
                    logger.error("系统当前处于无模型状态，请重新加载模型")
            else:
                logger.error("无法恢复，系统当前处于无模型状态，请重新加载模型")

            raise

