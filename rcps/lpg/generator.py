# rcps/lpg/generator.py (逻辑重构版)
from typing import Dict, List
from ..utils import get_logger
from ..agent import Agent

logger = get_logger(__name__)


class LayoutPrototypeGenerator:

    def __init__(self, layout_architect_agent: Agent):
        """
        Args:
            layout_architect_agent (Agent): 一个专门配置用于生成LDL的Agent。
        """
        self.agent = layout_architect_agent
        logger.info("LPG initialized using an LLM-based Layout Architect Agent.")

    def generate(self, slide_concept: Dict, slide_content: Dict) -> List[str]:
        logger.info(f"LPG (Agent-based) received slide concept and content.")

        ldl_sequence_json = self.agent.execute(
            slide_concept=slide_concept,
            slide_content=slide_content
        )

        # 假设LLM返回的JSON中有一个名为'ldl_sequence'的键
        ldl_sequence = ldl_sequence_json.get("ldl_sequence")
        if not ldl_sequence or not isinstance(ldl_sequence, list):
            logger.error(f"Layout Architect Agent returned invalid LDL sequence: {ldl_sequence_json}")
            # 提供一个安全的回退布局
            from .. import constants as C
            return [C.SOS_TOKEN, C.LAYOUT_CONTENT_ONE_COL, C.EOS_TOKEN]

        logger.info(f"Layout Architect Agent generated layout: {' '.join(ldl_sequence)}")
        return ldl_sequence