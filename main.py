# main.py
import yaml
from rcps.rcps_generator import RCPSGenerator
from rcps.utils import get_logger

logger = get_logger("RCPS_Main")


def main():
    try:
        # 加载主配置文件
        with open("config.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 初始化并运行生成器
        generator = RCPSGenerator(config)
        generator.run(
            pdf_path="./data/引言.pdf",  # 确保你的PDF文件放在这里
            output_path="./output/My_First_RCPS_Presentation.pptx"
        )

        logger.info("Presentation generation completed successfully!")

    except FileNotFoundError as e:
        logger.critical(f"Required file not found: {e}")
    except Exception as e:
        logger.critical(f"An error occurred during the main execution: {e}")


if __name__ == "__main__":
    main()