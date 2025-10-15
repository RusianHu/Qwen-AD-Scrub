"""
更新 transformers 库到最新版本
用于支持 Qwen3-VL 模型
"""

import subprocess
import sys


def update_transformers():
    """更新 transformers 到最新版本"""
    print("=" * 60)
    print("更新 transformers 库到最新版本")
    print("=" * 60)
    print()
    
    print("📦 正在卸载旧版本的 transformers...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "transformers", "-y"],
            check=False
        )
    except Exception as e:
        print(f"⚠ 卸载失败（可能未安装）: {e}")
    
    print()
    print("📦 正在安装最新版本的 transformers...")
    print("   (从 GitHub 源码安装,支持 Qwen3-VL)")
    print()
    
    try:
        # 使用国内镜像加速
        subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "git+https://github.com/huggingface/transformers.git",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ],
            check=True
        )
        
        print()
        print("=" * 60)
        print("✓ transformers 更新完成!")
        print("=" * 60)
        print()
        print("现在可以重新运行 app.py 启动应用")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("✗ 更新失败!")
        print("=" * 60)
        print()
        print("请手动运行以下命令:")
        print("  pip install git+https://github.com/huggingface/transformers.git")
        print()
        sys.exit(1)


if __name__ == "__main__":
    update_transformers()

