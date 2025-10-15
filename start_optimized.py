"""
使用优化配置启动 Qwen-AD-Scrub
自动启用 8-bit 量化和显存优化
"""

import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖是否安装"""
    logger.info("检查依赖...")
    
    try:
        import bitsandbytes
        logger.info("✓ bitsandbytes 已安装")
        return True
    except ImportError:
        logger.warning("✗ bitsandbytes 未安装")
        logger.info("正在安装 bitsandbytes...")
        
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "bitsandbytes>=0.41.0",
                "-i", "https://pypi.tuna.tsinghua.edu.cn/simple"
            ])
            logger.info("✓ bitsandbytes 安装成功")
            return True
        except subprocess.CalledProcessError:
            logger.error("✗ bitsandbytes 安装失败")
            logger.error("请手动安装: pip install bitsandbytes>=0.41.0")
            return False


def print_optimization_info():
    """打印优化信息"""
    logger.info("\n" + "="*60)
    logger.info("Qwen-AD-Scrub - 优化模式")
    logger.info("="*60)
    logger.info("\n已启用的优化:")
    logger.info("  ✓ 8-bit 量化 (节省 50% 显存)")
    logger.info("  ✓ 梯度检查点 (额外节省显存)")
    logger.info("  ✓ 降低默认采样率 (fps=0.3)")
    logger.info("  ✓ 降低默认分辨率 (64×28×28 像素)")
    logger.info("  ✓ 自动显存管理")
    logger.info("  ✓ 多卡支持 (自动分配)")
    logger.info("\n推荐设置:")
    logger.info("  - 短视频 (< 1分钟): fps=0.5-1.0, max_pixels=128*28*28")
    logger.info("  - 中等视频 (1-3分钟): fps=0.3-0.5, max_pixels=64*28*28")
    logger.info("  - 长视频 (> 3分钟): fps=0.2-0.3, max_pixels=64*28*28")
    logger.info("\n在 Web UI 中:")
    logger.info("  1. 初始化系统")
    logger.info("  2. ✅ 勾选 '使用 8-bit 量化'")
    logger.info("  3. 加载模型")
    logger.info("  4. 上传视频并分析")
    logger.info("="*60 + "\n")


def main():
    """主函数"""
    # 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，无法启动")
        sys.exit(1)
    
    # 打印优化信息
    print_optimization_info()
    
    # 启动应用
    logger.info("正在启动 Web UI...")
    logger.info("请在浏览器中访问: http://localhost:7860\n")
    
    try:
        # 导入并启动应用
        from app import create_ui
        
        app = create_ui()
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    except KeyboardInterrupt:
        logger.info("\n程序已停止")
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

