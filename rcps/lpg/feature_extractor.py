# rcps/lpg/feature_extractor.py
from typing import Dict, List
import numpy as np
from sentence_transformers import SentenceTransformer
from ..utils import get_logger

logger = get_logger(__name__)


class LPGFeatureExtractor:

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: str = 'cpu'):
        try:
            logger.info(f"Loading SentenceTransformer model: {model_name}...")
            self.text_model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.text_model.get_sentence_embedding_dimension()
            logger.info(f"SentenceTransformer model loaded successfully. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(
                f"Failed to load SentenceTransformer model '{model_name}'. Please ensure you have an internet connection for the first download. Error: {e}")
            self.text_model = None
            self.embedding_dim = 384  # Fallback dimension for all-MiniLM-L6-v2

    def extract(self, slide_plan: Dict) -> Dict:
        """
        从Planner生成的slide_plan中提取结构化的、可用于模型输入的slide_concept。
        """
        title = slide_plan.get("title", "")
        bullet_points = slide_plan.get("bullet_points", [])

        # 1. 数值和布尔特征 (Categorical & Numerical Features)
        concept = {
            "has_image": slide_plan.get("image_path") is not None,
            "bullet_points_count": len(bullet_points),
            "title_char_count": len(title),
            "body_char_count": sum(len(p) for p in bullet_points),
        }

        # 2. 语义特征 (Semantic Features)
        if self.text_model:
            full_text = title + ". " + ". ".join(bullet_points)
            if full_text.strip():
                # 使用真实模型进行编码
                embedding = self.text_model.encode(full_text, convert_to_numpy=True)
                concept['semantic_embedding'] = embedding
            else:
                # 如果没有文本，则使用零向量
                concept['semantic_embedding'] = np.zeros(self.embedding_dim)
        else:
            logger.warning("Text model not available. Semantic embedding will be a zero vector.")
            concept['semantic_embedding'] = np.zeros(self.embedding_dim)

        return concept