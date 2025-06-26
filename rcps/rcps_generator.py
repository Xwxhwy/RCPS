# rcps/rcps_generator.py (最终版)
from typing import Dict, List, Any

from .document import PDFParser
from RCPS_Project.rcps.lpg.lpg import LayoutPrototypeGenerator
from .agent import AgentFactory
from .presentation import Presentation, SlidePage, Shape
from .renderer import PptxRenderer
from .utils import get_logger, ensure_dir
from .exceptions import RCPSException, ConfigError
from . import constants as C

logger = get_logger(__name__)


class RCPSGenerator:


    def __init__(self, config: Dict[str, Any]):
        """
        初始化RCPS生成器。

        Args:
            config (Dict[str, Any]): 一个包含所有配置的字典。

        Raises:
            ConfigError: 如果配置不完整或不正确。
        """
        self.config = config
        self.workspace_dir = config.get("workspace_dir", "workspace")
        ensure_dir(self.workspace_dir)
        logger.info(f"RCPS Generator initializing with workspace: {self.workspace_dir}")

        try:
            self.pdf_parser = PDFParser(workspace_dir=self.workspace_dir)


            llm_map = self._initialize_llms(config['llms'])
            self.agent_factory = AgentFactory(llm_map)

            self.lpg = LayoutPrototypeGenerator(
                model_path=config['lpg']['model_path'],
                use_mock=config['lpg'].get('use_mock', False)
            )

            self.renderer = PptxRenderer()

        except KeyError as e:
            raise ConfigError(f"Configuration is missing a required key: {e}")
        except Exception as e:
            logger.error(f"Error during RCPSGenerator initialization: {e}", exc_info=True)
            raise RCPSException("Failed to initialize RCPS components.") from e

        logger.info("RCPS Generator initialized successfully.")

    def _initialize_llms(self, llm_configs: List[Dict]) -> Dict[str, Any]:
        from .llms import OpenAICompatibleLLM

        llm_map = {}
        for conf in llm_configs:
            name = conf['name']
            llm_map[name] = OpenAICompatibleLLM(
                model=conf['model'],
                api_key=conf['api_key'],
                base_url=conf['base_url']
            )
            logger.info(f"Initialized LLM '{name}' with model '{conf['model']}'.")
        return llm_map

    def run(self, pdf_path: str, output_path: str) -> Presentation:
        try:
            # ==================================================================
            # 阶段一：文档解析与叙事规划 (R-CoT)
            # ==================================================================
            logger.info("--- STAGE 1: Document Parsing and Narrative Planning (R-CoT) ---")
            doc = self.pdf_parser.parse(pdf_path)

            planner_agent = self.agent_factory.get_agent(C.AGENT_PLANNER)
            narrative_plan = planner_agent.execute(document_text=doc.get_full_text())
            logger.info(f"Narrative plan generated with {len(narrative_plan)} slides.")

            presentation = Presentation(
                presentation_id=doc.doc_id,
                design_style=self.config.get('design_style', C.DESIGN_STYLE_MODERN),
                color_palette=self.config.get('color_palette', C.COLOR_PALETTE_CORPORATE_BLUE)
            )

            content_agent = self.agent_factory.get_agent(C.AGENT_CONTENT_GENERATOR)

            for i, slide_plan in enumerate(narrative_plan):
                page_id = i + 1
                logger.info(
                    f"--- Processing Slide {page_id}/{len(narrative_plan)}: {slide_plan.get('purpose', '')} ---")

                # ==================================================================
                # 阶段二：布局生成与内容实例化
                # ==================================================================
                logger.info(f"[Slide {page_id}] STAGE 2: Layout & Content Generation")


                slide_concept = slide_plan.get('concept', {})
                layout_tokens = self.lpg.generate(slide_concept=slide_concept)


                source_chunk = slide_plan.get('source_text_chunk', '')
                slide_content_details = content_agent.execute(
                    slide_purpose=slide_plan.get('purpose', ''),
                    source_text_chunk=source_chunk
                )

                slide_page = SlidePage.from_plan_and_layout(page_id, slide_content_details, layout_tokens)

                # ==================================================================
                # 阶段三：迭代多模态优化 (IMR)
                # ==================================================================
                logger.info(f"[Slide {page_id}] STAGE 3: Iterative Multi-modal Refinement (IMR)")
                slide_page = self._run_imr_loop(slide_page)

                presentation.add_slide(slide_page)

            # --- 工作流结束：渲染最终演示文稿 ---
            logger.info("--- WORKFLOW COMPLETE: Rendering final presentation ---")
            self.renderer.render(presentation, output_path)

            return presentation

        except RCPSException as e:
            logger.critical(f"A critical error occurred in the RCPS pipeline: {e}", exc_info=True)
            raise  # 重新引发，让上层调用者知道失败了
        except Exception as e:
            logger.critical(f"An unexpected error terminated the RCPS pipeline: {e}", exc_info=True)
            raise RCPSException("An unexpected error occurred.") from e

    def _run_imr_loop(self, slide_page: SlidePage) -> SlidePage:
        """
        执行IMR循环，对单张幻灯片进行迭代优化。
        """
        critic_agent = self.agent_factory.get_agent(C.AGENT_CRITIC)
        refiner_agent = self.agent_factory.get_agent(C.AGENT_REFINER)

        max_cycles = self.config.get("imr_cycles", 2)

        for cycle in range(max_cycles):
            logger.info(f"[Slide {slide_page.page_id}] IMR Cycle {cycle + 1}/{max_cycles}")


            current_state_dict = [shape.__dict__ for shape in slide_page.shapes]

            critiques = critic_agent.execute(
                slide_title=slide_page.title,
                slide_elements_json=current_state_dict
            )

            if not critiques or critiques.get("status") == C.IMR_STATUS_NO_CHANGE:
                logger.info(f"[Slide {slide_page.page_id}] Critic found no major issues. Concluding IMR loop.")
                break

            logger.debug(f"[Slide {slide_page.page_id}] Critic suggestions: {critiques.get('suggestions')}")


            refined_elements_json = refiner_agent.execute(
                original_elements_json=current_state_dict,
                critiques=critiques.get('suggestions', [])
            )

            try:
                # 使用新的数据更新Shape对象
                new_shapes = [Shape(**elem_data) for elem_data in refined_elements_json]
                slide_page.shapes = new_shapes
                logger.info(f"[Slide {slide_page.page_id}] Slide has been refined based on critiques.")
            except (TypeError, KeyError) as e:
                logger.error(
                    f"[Slide {slide_page.page_id}] Refiner agent returned invalid data. Skipping refinement for this cycle. Error: {e}")

        else:  # for...else 循环，只有当for循环正常结束（未被break）时执行
            logger.info(f"[Slide {slide_page.page_id}] Reached max IMR cycles.")

        return slide_page