from typing import Dict, List
import torch 


from RCPS_Project.rcps import constants as C
# -----------------------------------------------

from RCPS_Project.rcps.utils import get_logger
from RCPS_Project.rcps.exceptions import LayoutError

logger = get_logger(__name__)


class LayoutPrototypeGenerator:
    def __init__(self, model_path: str, device: str = 'cpu', use_mock: bool = False):
        self.use_mock = use_mock
        if self.use_mock:
            self.model = None
            logger.warning("LPG is running in MOCK mode. It will return a fixed, rule-based layout.")
            return

        self.device = torch.device(device)
        try:

            raise NotImplementedError("LPG model loading is not implemented yet.")
        except FileNotFoundError:
            logger.error(f"LPG model file not found at: {model_path}")
            raise LayoutError(f"LPG model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load LPG model from {model_path}: {e}")
            raise LayoutError("Failed to load LPG model.") from e

    def _concept_to_tensor(self, slide_concept: Dict) -> torch.Tensor:
        num_points = slide_concept.get("bullet_points_count", 0)
        has_image = 1.0 if slide_concept.get("has_image", False) else 0.0
        title_len = len(slide_concept.get("title_suggestion", ""))
        body_len = len(slide_concept.get("body_suggestion", ""))

        # 将特征归一化或嵌入，然后拼接成一个向量
        feature_vector = [num_points / 10.0, has_image, title_len / 50.0, body_len / 500.0]
        return torch.tensor([feature_vector], dtype=torch.float32).to(self.device)

    def generate(self, slide_concept: Dict) -> List[str]:

        logger.info(f"LPG received slide concept: {slide_concept}")

        if self.use_mock:
            has_image = slide_concept.get("has_image", False)
            num_points = slide_concept.get("bullet_points_count", 0)


            layout = [C.SOS_TOKEN]

            if has_image:
                layout.extend([C.LAYOUT_CONTENT_TWO_COL, C.SEP_TOKEN, C.ELEM_TITLE, C.POS_TOP, C.SEP_TOKEN])
                if num_points > 3:
                    # 如果文本多，文本占主要位置
                    layout.extend([C.ELEM_BODY_TEXT, C.POS_LEFT, C.ATTR_SIZE_MAJOR, C.SEP_TOKEN, C.ELEM_IMAGE, C.POS_RIGHT, C.ATTR_SIZE_MINOR])
                else:
                    # 如果图片是重点
                    layout.extend([C.ELEM_IMAGE, C.POS_LEFT, C.ATTR_SIZE_MAJOR, C.SEP_TOKEN, C.ELEM_BODY_TEXT, C.POS_RIGHT, C.ATTR_SIZE_MINOR])
            else:
                layout.extend([C.LAYOUT_CONTENT_ONE_COL, C.SEP_TOKEN, C.ELEM_TITLE, C.POS_TOP, C.SEP_TOKEN, C.ELEM_BODY_TEXT, C.POS_MIDDLE])

            layout.append(C.EOS_TOKEN)
            # ------------------------------------

            logger.info(f"Mock LPG generated layout: {' '.join(layout)}")
            return layout


        try:
            with torch.no_grad():
                input_tensor = self._concept_to_tensor(slide_concept)

                raise NotImplementedError("LPG real model inference is not implemented yet.")

                # logger.info(f"LPG model generated layout: {' '.join(ldl_tokens)}")
                # return ldl_tokens
        except Exception as e:
            logger.error(f"Error during LPG model inference: {e}")
            raise LayoutError("LPG model inference failed.") from e
