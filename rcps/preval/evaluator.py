# rcps/preval/evaluator.py

from typing import Dict, List, Any
import torch
import torch.nn as nn
import numpy as np
import os

from .feature_extractor import PrevalFeatureExtractor
from ..utils import get_logger

logger = get_logger(__name__)


class PrevalPreferenceModel(nn.Module):
    """
    一个基于MLP的偏好模型，用于PREVAL。

    该模型接收一个扁平化的特征向量，并输出三个维度的质量分数：
    Content, Coherence, 和 Design。
    这对应了论文中描述的“multi-dimensional quality assessment models”。
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int = 3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 将分数约束在[0, 1]区间

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PREVAL:
    """
    [严谨实现] PREVAL评估器。
    加载一个预训练的、基于MLP的偏好模型来评估演示文稿的质量。
    """
    # 定义特征的固定顺序，以确保特征向量的一致性
    FEATURE_ORDER = [
        'text_coherence', 'avg_text_length', 'text_redundancy',
        'avg_alignment_score', 'avg_balance_score', 'avg_whitespace_ratio',
        'avg_aesthetic_score', 'avg_image_text_alignment'
        # ... 可以添加更多来自feature_extractor的特征
    ]

    def __init__(self, model_path: str, model_config: Dict, device: str = 'cpu'):
        """
        Args:
            model_path (str): 预训练的PREVAL模型权重文件 (.pt) 的路径。
            model_config (Dict): 包含模型超参数的字典 (input_dim, hidden_dims)。
            device (str): 运行设备的字符串 ('cpu' or 'cuda')。
        """
        self.device = torch.device(device)
        self.feature_extractor = PrevalFeatureExtractor(device=self.device)

        # 1. 实例化一个结构确定的模型
        self.model = PrevalPreferenceModel(
            input_dim=model_config['input_dim'],
            hidden_dims=model_config['hidden_dims'],
            output_dim=3
        ).to(self.device)

        # 2. 从指定路径加载预训练的权重
        try:
            logger.info(f"Loading PREVAL model state_dict from '{model_path}'...")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            logger.info("PREVAL model loaded successfully.")
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. PREVAL cannot function.")
            raise
        except Exception as e:
            logger.error(f"Failed to load PREVAL model from {model_path}: {e}")
            raise

    def _create_feature_vector(self, features: Dict[str, Any]) -> torch.Tensor:
        """将特征字典转换为一个固定顺序的特征向量。"""
        vector = [features.get(key, 0.0) for key in self.FEATURE_ORDER]
        return torch.tensor(vector, dtype=torch.float32).to(self.device)

    def evaluate(self, pptx_path: str) -> Dict[str, float]:
        """
        评估一个PPTX文件，并返回多维度分数。
        """
        # 1. 提取特征
        features = self.feature_extractor.extract(pptx_path)
        if not features:
            logger.error(f"Feature extraction failed for '{pptx_path}'.")
            return {"Error": 1.0, "Message": "Failed to extract features."}

        # 2. 将特征转换为模型的输入张量
        feature_vector = self._create_feature_vector(features)

        # 3. 执行模型推理
        with torch.no_grad():
            scores_tensor = self.model(feature_vector.unsqueeze(0)).squeeze(0)

        # 4. 格式化输出
        content_score, coherence_score, design_score = scores_tensor.cpu().numpy()
        overall_score = np.mean([content_score, coherence_score, design_score])

        scores = {
            "Content": round(float(content_score), 4),
            "Coherence": round(float(coherence_score), 4),
            "Design": round(float(design_score), 4),
            "Overall": round(float(overall_score), 4),
        }
        logger.info(f"PREVAL scores for '{pptx_path}': {scores}")
        return scores

