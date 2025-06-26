# rcps/utils.py
import logging
import os
import json
import re
from typing import Dict, Any, List, Union

from tenacity import retry, stop_after_attempt, wait_fixed, before_sleep_log
from .exceptions import ParsingError

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """获取一个配置好的日志记录器。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

_logger_for_retry = get_logger("tenacity_retry")

tenacity_retry = retry(
    wait=wait_fixed(3),
    stop=stop_after_attempt(3),
    before_sleep=before_sleep_log(_logger_for_retry, logging.WARNING),
    reraise=True
)

def package_join(*paths: str) -> str:
    """获取相对于项目根目录的路径。"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, *paths)

def get_json_from_response(response: str) -> Union[Dict, List]:
    """从LLM返回的文本中稳健地提取JSON。"""
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ParsingError(f"Found JSON block, but failed to parse: {e}") from e

    start_brace = response.find('{')
    end_brace = response.rfind('}')
    if start_brace != -1 and end_brace > start_brace:
        json_str = response[start_brace : end_brace + 1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        raise ParsingError(f"Could not parse valid JSON from response.") from e

def ensure_dir(path: str):
    """确保目录存在，如果不存在则创建。"""
    os.makedirs(path, exist_ok=True)