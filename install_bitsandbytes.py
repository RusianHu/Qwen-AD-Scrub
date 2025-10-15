"""
安装 bitsandbytes 库
用于 8-bit 量化，节省显存
"""

import subprocess
import sys


def install_bitsandbytes():
    """安装 bitsandbytes"""
    print("正在安装 bitsandbytes...")
    print("这将用于 8-bit 量化，可节省约 50% 的显存")
    
    try:
        # 使用国内镜像源加速下载
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "bitsandbytes>=0.41.0",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
        ])
        
        print("\n✓ bitsandbytes 安装成功!")
        print("\n现在可以使用 8-bit 量化来节省显存了。")
        print("在 Web UI 中勾选 '使用 8-bit 量化' 选项即可。")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 安装失败: {e}")
        print("\n请尝试手动安装:")
        print("  pip install bitsandbytes>=0.41.0")
        return False


if __name__ == "__main__":
    install_bitsandbytes()

