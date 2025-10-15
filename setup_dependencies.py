"""
依赖检查和下载脚本
用于检查和下载 Qwen3-VL 系列模型和 FFmpeg
支持多个不同参数量的模型
"""

import os
import sys
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm
import subprocess
import shutil
from config import SUPPORTED_MODELS


class DependencyManager:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        self.ffmpeg_dir = self.project_root / "FFmpeg"

        # 支持的模型配置
        self.supported_models = SUPPORTED_MODELS
        
    def check_model(self, model_name=None):
        """
        检查模型文件是否存在且完整

        Args:
            model_name: 模型名称，如果为 None 则检查所有支持的模型

        Returns:
            如果指定了 model_name，返回该模型是否完整
            如果未指定，返回已安装模型的列表
        """
        if model_name is not None:
            # 检查特定模型
            if model_name not in self.supported_models:
                print(f"✗ 不支持的模型: {model_name}")
                return False

            model_path = self.supported_models[model_name]['model_path']
            return self._check_single_model(model_name, model_path)
        else:
            # 检查所有模型
            installed_models = []
            print("\n" + "=" * 60)
            print("检查已安装的模型")
            print("=" * 60)

            for name, config in self.supported_models.items():
                model_path = config['model_path']
                if self._check_single_model(name, model_path, verbose=False):
                    installed_models.append(name)
                    print(f"✓ {config['display_name']}")

            if not installed_models:
                print("\n未找到任何已安装的模型")
            else:
                print(f"\n共找到 {len(installed_models)} 个已安装的模型")

            return installed_models

    def _check_single_model(self, model_name, model_path, verbose=True):
        """检查单个模型文件是否完整"""
        if not model_path.exists():
            if verbose:
                print(f"✗ 模型目录不存在: {model_name}")
            return False

        # 检查关键配置文件
        required_config_files = [
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json"
        ]

        for file in required_config_files:
            if not (model_path / file).exists():
                if verbose:
                    print(f"✗ 缺少配置文件: {file}")
                return False

        # 检查模型权重文件
        # 读取 index 文件获取需要的权重文件列表
        import json
        try:
            with open(model_path / "model.safetensors.index.json", 'r', encoding='utf-8') as f:
                index_data = json.load(f)

            # 获取所有需要的权重文件
            weight_files = set(index_data.get("weight_map", {}).values())

            missing_weights = []
            for weight_file in weight_files:
                if not (model_path / weight_file).exists():
                    missing_weights.append(weight_file)

            if missing_weights:
                if verbose:
                    print(f"✗ 缺少 {len(missing_weights)} 个模型权重文件:")
                    for wf in missing_weights[:3]:  # 只显示前3个
                        print(f"  - {wf}")

                    # 检查是否有未完成的下载（提示用户可以继续）
                    cache_dir = model_path / ".cache" / "huggingface" / "download"
                    if cache_dir.exists():
                        incomplete_files = list(cache_dir.glob("*.incomplete"))
                        if incomplete_files:
                            total_size = sum(f.stat().st_size for f in incomplete_files) / (1024 * 1024)
                            print(f"\n💡 检测到未完成的下载 (已下载 {total_size:.1f} MB)")
                            print("   可以继续下载以完成模型安装")

                return False

            # 检查文件大小是否合理（不为0）
            for weight_file in weight_files:
                file_path = model_path / weight_file
                if file_path.stat().st_size == 0:
                    if verbose:
                        print(f"✗ 模型权重文件大小为0: {weight_file}")
                    return False

            if verbose:
                print(f"✓ 模型文件完整: {model_path}")
                print(f"  - 配置文件: {len(required_config_files)} 个")
                print(f"  - 权重文件: {len(weight_files)} 个")
            return True

        except Exception as e:
            if verbose:
                print(f"✗ 检查模型文件时出错: {e}")
            return False
    
    def download_model(self, model_name=None):
        """
        使用 huggingface_hub 库从镜像站下载模型

        Args:
            model_name: 要下载的模型名称，如果为 None 则提示用户选择
        """
        # 如果未指定模型，让用户选择
        if model_name is None:
            print("\n" + "=" * 60)
            print("可用的模型:")
            print("=" * 60)

            for i, (name, config) in enumerate(self.supported_models.items(), 1):
                print(f"{i}. {config['display_name']}")
                print(f"   描述: {config['description']}")
                print(f"   最小显存: {config['min_vram_gb']}GB (8-bit 量化)")
                print(f"   推荐显存: {config['recommended_vram_gb']}GB")
                print()

            try:
                choice = int(input("请选择要下载的模型 (输入序号): "))
                model_name = list(self.supported_models.keys())[choice - 1]
            except (ValueError, IndexError):
                print("✗ 无效的选择")
                return False

        # 验证模型名称
        if model_name not in self.supported_models:
            print(f"✗ 不支持的模型: {model_name}")
            return False

        model_config = self.supported_models[model_name]
        model_path = model_config['model_path']
        hf_model_id = model_config['hf_model_id']

        print("\n" + "=" * 60)
        print(f"开始下载 {model_config['display_name']}")
        print("=" * 60)
        print(f"描述: {model_config['description']}")
        print(f"使用 HF-Mirror 镜像站加速下载")
        print("请耐心等待...")

        # 检查是否有未完成的下载
        cache_dir = model_path / ".cache" / "huggingface" / "download"

        if cache_dir.exists():
            incomplete_files = list(cache_dir.glob("*.incomplete"))
            if incomplete_files:
                total_size = sum(f.stat().st_size for f in incomplete_files) / (1024 * 1024)
                print(f"\n💡 检测到 {len(incomplete_files)} 个未完成的下载 (已下载 {total_size:.1f} MB)")
                print("   将自动从断点继续下载...")

        print()

        # 创建模型目录
        self.models_dir.mkdir(exist_ok=True)

        # 设置环境变量使用镜像站
        hf_mirror = model_config['hf_mirror']
        os.environ['HF_ENDPOINT'] = hf_mirror

        try:
            # 检查 huggingface_hub 是否已安装
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                print("✗ 未找到 huggingface_hub 库")
                print("\n请先安装 huggingface_hub:")
                print("  pip install -U huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
                print("\n安装完成后，重新运行此脚本")
                return False

            print("使用 huggingface_hub 库下载（自动支持断点续传）")
            print()

            # 使用 snapshot_download 下载整个模型仓库
            # 这个函数会自动处理断点续传、并发下载等
            try:
                snapshot_download(
                    repo_id=hf_model_id,
                    local_dir=str(model_path),
                    local_dir_use_symlinks=False,  # 直接下载文件，不使用符号链接
                    resume_download=True,  # 启用断点续传
                    max_workers=8,  # 并发下载线程数
                )

                print("\n" + "=" * 60)
                print(f"✓ 模型 {model_name} 下载完成!")
                print("=" * 60)

                # 再次检查模型完整性
                if self.check_model(model_name):
                    return True
                else:
                    print("\n⚠ 警告: 下载完成但模型文件不完整，请重新运行脚本")
                    return False

            except KeyboardInterrupt:
                print("\n\n⚠ 下载被用户中断")
                print("💡 提示: 重新运行脚本将自动从断点继续下载")
                return False
            except Exception as e:
                print(f"\n✗ 下载过程中出错: {e}")
                print("\n💡 提示: 重新运行脚本将自动从断点继续下载")
                return False

        except Exception as e:
            print(f"\n✗ 下载过程中出错: {e}")
            print("\n你可以手动下载模型:")
            print("  1. 安装 huggingface_hub:")
            print("     pip install -U huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple")
            print()
            print("  2. 使用 Python 下载:")
            print("     python -c \"import os; os.environ['HF_ENDPOINT']='https://hf-mirror.com'; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_dir='models/Qwen3-VL-4B-Instruct', local_dir_use_symlinks=False, resume_download=True)\"")
            print()
            print("  或使用命令行工具:")
            print("     $env:HF_ENDPOINT='https://hf-mirror.com'  # PowerShell")
            print("     hf download Qwen/Qwen3-VL-4B-Instruct --local-dir models/Qwen3-VL-4B-Instruct")
            print()
            print("  💡 提示: 如果下载中断，直接重新运行相同命令即可继续下载")
            return False
    
    def check_ffmpeg(self):
        """检查 FFmpeg 是否存在"""
        # 动态查找 FFmpeg 可执行文件
        ffmpeg_exe = self._find_ffmpeg_exe()

        if ffmpeg_exe and ffmpeg_exe.exists():
            # 验证 FFmpeg 可执行
            try:
                result = subprocess.run(
                    [str(ffmpeg_exe), "-version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    # 提取版本信息
                    version_line = result.stdout.split('\n')[0]
                    print(f"✓ FFmpeg 已就绪: {ffmpeg_exe}")
                    print(f"  版本: {version_line}")
                    return True
                else:
                    print(f"✗ FFmpeg 文件存在但无法执行: {ffmpeg_exe}")
                    return False
            except Exception as e:
                print(f"✗ FFmpeg 验证失败: {e}")
                return False

        # 检查是否有压缩包
        zip_file = self.ffmpeg_dir / "ffmpeg-release-essentials.zip"
        if zip_file.exists():
            size_mb = zip_file.stat().st_size / (1024 * 1024)
            print(f"✗ FFmpeg 未安装")
            print(f"  发现压缩包: {zip_file.name} ({size_mb:.1f} MB)")
            print(f"  需要解压安装")
            return False

        print("✗ FFmpeg 未安装，也未找到压缩包")
        return False

    def _find_ffmpeg_exe(self):
        """动态查找 FFmpeg 可执行文件"""
        if not self.ffmpeg_dir.exists():
            return None

        # 优先检查标准位置 /FFmpeg/bin/ffmpeg.exe
        standard_exe = self.ffmpeg_dir / "bin" / "ffmpeg.exe"
        if standard_exe.exists():
            return standard_exe

        # 查找所有包含 ffmpeg 的文件夹
        for item in self.ffmpeg_dir.iterdir():
            if item.is_dir() and "ffmpeg" in item.name.lower():
                bin_dir = item / "bin"
                if bin_dir.exists():
                    exe = bin_dir / "ffmpeg.exe"
                    if exe.exists():
                        return exe

        return None
    
    def extract_ffmpeg(self):
        """解压 FFmpeg 并重新组织目录结构"""
        zip_file = self.ffmpeg_dir / "ffmpeg-release-essentials.zip"

        if not zip_file.exists():
            print("✗ FFmpeg 压缩包不存在，开始下载...")
            return self.download_ffmpeg()

        print(f"\n开始解压 FFmpeg: {zip_file}")

        try:
            # 解压到临时目录
            temp_dir = self.ffmpeg_dir / "temp_extract"
            temp_dir.mkdir(exist_ok=True)

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # 获取总文件数用于进度条
                total_files = len(zip_ref.namelist())

                with tqdm(total=total_files, desc="解压中", unit="文件") as pbar:
                    for file in zip_ref.namelist():
                        zip_ref.extract(file, temp_dir)
                        pbar.update(1)

            # 查找解压后的 FFmpeg 文件夹
            extracted_folder = None
            for item in temp_dir.iterdir():
                if item.is_dir() and "ffmpeg" in item.name.lower():
                    extracted_folder = item
                    break

            if not extracted_folder:
                print("✗ 未找到解压后的 FFmpeg 文件夹")
                shutil.rmtree(temp_dir)
                return False

            # 重新组织目录结构：将 bin、doc、presets 等移动到 FFmpeg 根目录
            print("重新组织目录结构...")

            # 移动 bin 目录
            src_bin = extracted_folder / "bin"
            dst_bin = self.ffmpeg_dir / "bin"
            if src_bin.exists():
                if dst_bin.exists():
                    shutil.rmtree(dst_bin)
                shutil.move(str(src_bin), str(dst_bin))
                print(f"✓ 已移动 bin 目录到: {dst_bin}")

            # 移动其他文件（可选）
            for item in ["doc", "presets", "LICENSE", "README.txt"]:
                src_item = extracted_folder / item
                if src_item.exists():
                    dst_item = self.ffmpeg_dir / item
                    if dst_item.exists():
                        if dst_item.is_dir():
                            shutil.rmtree(dst_item)
                        else:
                            dst_item.unlink()
                    shutil.move(str(src_item), str(dst_item))

            # 清理临时目录
            shutil.rmtree(temp_dir)

            print("✓ FFmpeg 解压并重组完成!")
            print(f"✓ FFmpeg 可执行文件位于: {self.ffmpeg_dir / 'bin' / 'ffmpeg.exe'}")
            return True

        except Exception as e:
            print(f"✗ 解压失败: {e}")
            # 清理临时目录
            if (self.ffmpeg_dir / "temp_extract").exists():
                shutil.rmtree(self.ffmpeg_dir / "temp_extract")
            return False
    
    def download_ffmpeg(self):
        """下载 FFmpeg"""
        url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
        zip_file = self.ffmpeg_dir / "ffmpeg-release-essentials.zip"
        
        print(f"\n开始下载 FFmpeg: {url}")
        
        try:
            # 创建目录
            self.ffmpeg_dir.mkdir(exist_ok=True)
            
            # 下载文件
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(zip_file, 'wb') as f, tqdm(
                desc="下载中",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print("✓ FFmpeg 下载完成!")
            return self.extract_ffmpeg()
            
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            print("\n请手动下载 FFmpeg:")
            print(f"  下载地址: {url}")
            print(f"  保存到: {zip_file}")
            return False
    
    def get_ffmpeg_path(self):
        """获取 FFmpeg 可执行文件路径"""
        ffmpeg_exe = self._find_ffmpeg_exe()
        return str(ffmpeg_exe) if ffmpeg_exe else None
    
    def setup_all(self):
        """检查并设置所有依赖"""
        print("=" * 60)
        print("Qwen-AD-Scrub 依赖检查工具")
        print("=" * 60)
        print()

        # 检查模型
        print("[1/2] 检查模型文件...")
        print("-" * 60)
        model_ok = self.check_model()

        if not model_ok:
            print("\n模型文件不存在或不完整")
            choice = input("\n是否下载模型? (y/n): ").strip().lower()
            if choice == 'y':
                model_ok = self.download_model()
            else:
                print("跳过模型下载")

        print()

        # 检查 FFmpeg
        print("[2/2] 检查 FFmpeg...")
        print("-" * 60)
        ffmpeg_ok = self.check_ffmpeg()

        if not ffmpeg_ok:
            print("\nFFmpeg 未安装")
            choice = input("\n是否解压/下载 FFmpeg? (y/n): ").strip().lower()
            if choice == 'y':
                ffmpeg_ok = self.extract_ffmpeg()
            else:
                print("跳过 FFmpeg 设置")

        # 显示最终结果
        print("\n" + "=" * 60)
        print("依赖检查完成!")
        print("=" * 60)

        # 再次检查状态
        final_model_ok = self.check_model()
        final_ffmpeg_ok = self.check_ffmpeg()

        print("\n📋 依赖状态:")
        print(f"  模型 (Qwen3-VL-4B-Instruct): {'✓ 已就绪' if final_model_ok else '✗ 未就绪'}")
        print(f"  FFmpeg:                      {'✓ 已就绪' if final_ffmpeg_ok else '✗ 未就绪'}")

        if final_model_ok and final_ffmpeg_ok:
            print("\n✓ 所有依赖已就绪! 可以运行程序了。")
            ffmpeg_path = self.get_ffmpeg_path()
            if ffmpeg_path:
                print(f"\n📁 FFmpeg 路径: {ffmpeg_path}")
            print("\n🚀 运行程序:")
            print("   python app.py")
        else:
            print("\n⚠ 部分依赖缺失:")
            if not final_model_ok:
                print("  - 模型文件未就绪，请重新运行脚本下载")
            if not final_ffmpeg_ok:
                print("  - FFmpeg 未就绪，请重新运行脚本安装")

        print()


def main():
    """主函数 - 交互式菜单"""
    print("=" * 60)
    print("Qwen-AD-Scrub 依赖检查和下载工具")
    print("=" * 60)

    manager = DependencyManager()

    # 检查已安装的模型
    print("\n1. 检查已安装的模型...")
    installed_models = manager.check_model()

    # 检查 FFmpeg
    print("\n2. 检查 FFmpeg...")
    ffmpeg_ok = manager.check_ffmpeg()

    # 显示菜单
    while True:
        print("\n" + "=" * 60)
        print("主菜单")
        print("=" * 60)
        print("1. 下载新模型")
        print("2. 检查已安装的模型")
        print("3. 下载/检查 FFmpeg")
        print("4. 退出")

        choice = input("\n请选择 (1-4): ").strip()

        if choice == "1":
            # 下载新模型
            manager.download_model()
            # 重新检查已安装的模型
            installed_models = manager.check_model()

        elif choice == "2":
            # 检查已安装的模型
            installed_models = manager.check_model()

        elif choice == "3":
            # 检查/下载 FFmpeg
            if not manager.check_ffmpeg():
                print("\n是否下载 FFmpeg? (y/n): ", end="")
                if input().strip().lower() == 'y':
                    manager.download_ffmpeg()

        elif choice == "4":
            # 退出
            if installed_models and ffmpeg_ok:
                print("\n" + "=" * 60)
                print("✓ 所有依赖已就绪!")
                print("=" * 60)
                print(f"\n已安装 {len(installed_models)} 个模型:")
                for model in installed_models:
                    print(f"  - {model}")
                print("\n你可以运行以下命令启动应用:")
                print("  python app.py")
                print("\n或使用优化启动脚本:")
                print("  python start_optimized.py")
            else:
                print("\n提示: 还有依赖未安装，建议先完成安装")
            break
        else:
            print("✗ 无效的选择，请重试")

    print("\n按 Enter 键退出...")
    input()


if __name__ == "__main__":
    # 使用新的交互式菜单
    main()

