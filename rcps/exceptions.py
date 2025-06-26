

class RCPSException(Exception):
    """RCPS项目所有自定义异常的基类。"""
    def __init__(self, message="An error occurred in the RCPS framework."):
        self.message = message
        super().__init__(self.message)

class LLMError(RCPSException):
    """当与大语言模型(LLM)的交互过程中发生错误时引发。"""
    def __init__(self, message="An error occurred during interaction with the LLM."):
        super().__init__(message)

class ParsingError(RCPSException):
    """当解析输入文件（如PDF）或模型输出（如JSON）失败时引发。"""
    def __init__(self, message="Failed to parse a file or model output."):
        super().__init__(message)

class GenerationError(RCPSException):
    """在演示文稿生成的核心逻辑中发生不可恢复的错误时引发。"""
    def __init__(self, message="A critical error occurred during presentation generation."):
        super().__init__(message)

class LayoutError(RCPSException):
    """与布局生成或解释相关的错误。"""
    def __init__(self, message="An error occurred related to layout generation or interpretation."):
        super().__init__(message)

class ConfigError(RCPSException):
    """当配置文件缺失或格式不正确时引发。"""
    def __init__(self, message="Configuration error."):
        super().__init__(message)