# RCPS: Reflective Coherent Presentation Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

This is the official PyTorch implementation for the paper: **"Multi-Agent Synergy-Driven Iterative Visual Narrative Synthesis"**.

RCPS is a novel framework designed to automate the generation of high-quality presentations. By leveraging a multi-agent system, RCPS produces logically coherent, content-adaptive, and visually compelling slides from source documents.

## Key Features

-   **Deep Structured Narrative Planning (R-CoT)**: Ensures logical flow and coherence across the presentation.
-   **Adaptive Layout Generation (LPG)**: A trainable module that generates symbolic layouts sensitive to content.
-   **Iterative Multi-Modal Refinement (IMR)**: A feedback loop that progressively improves slide quality.
-   **PREVAL Evaluation Framework**: A novel, preference-based framework for accurately assessing presentation quality.

## Release Information

This repository provides the core implementation of the RCPS framework to ensure the reproducibility of our research.

**Current Version: `v1.0.0` (Initial Release)**

This initial release is intended to accompany our paper submission and includes:
-   ✔️ The complete training pipeline for the Layout Prototype Generator (LPG).
-   ✔️ A functional implementation of the PREVAL evaluator.
-   ✔️ The core modules for LDL, layout interpretation, and rendering logic.

This codebase is under **active development**. We are in the process of organizing and uploading additional components, including pre-trained models and end-to-end inference scripts. We are committed to improving its structure, documentation, and functionality.

## Key Results and Analysis

This section presents further analysis and visualizations to provide deeper insights into the performance of our framework, directly addressing key evaluation aspects.

### 1. Diverse, Multi-Domain Dataset

> To ensure the robustness and generalizability of our framework, RCPS is trained and evaluated on a diverse dataset spanning multiple domains. This prevents overfitting to a single field and validates its applicability to a wide range of topics.

![Dataset Distribution](assets/chart_domain_distribution.png)

### 2. Layout Performance: Adaptive vs. Fixed Templates

> A core contribution of RCPS is its **Adaptive Layout Generation (LPG)**. We conducted a comparative analysis against a fixed-template approach, similar to that used by systems like PPTAgent. The results clearly show that while templates are adequate for simple slides, their performance degrades sharply as content complexity increases. In contrast, **our LPG maintains high layout quality**, proving its superior flexibility and robustness for real-world scenarios.

![Layout Performance Comparison](assets/chart_layout_performance.png)

### 3. Head-to-Head Comparison with SOTA (PPTAgent)

> To directly benchmark against the state-of-the-art, we performed a head-to-head pairwise preference study comparing RCPS with PPTAgent. Human evaluators showed a strong preference for our method, particularly in the critical dimensions of **Design** and **Overall Quality**. This quantitatively validates the effectiveness of our framework's core design principles.

![RCPS vs PPTAgent Performance](assets/chart_rcps_vs_pptagent.png)

### 4. Impact of Training Data Distribution

> We analyzed how performance correlates with the amount of in-domain training data. As expected, RCPS exhibits stronger performance and faster convergence in data-rich domains (e.g., Education). Importantly, even in data-poor domains (e.g., Arts & Design), **our model still significantly outperforms the baseline**, showcasing its effective and generalizable learning capabilities.

![Model Performance by Domain](assets/chart_model_performance_by_domain.png)

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/rcps-project.git
    cd rcps-project
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Getting Started

A quick example of how to run the LPG training pipeline:

1.  **Prepare Data**: Use `python -m training.lpg_training.build_dataset` to process your PPTX files into a `.jsonl` dataset.
2.  **Build Tokenizer**: Create a vocabulary from your dataset using the `LDLTokenizer`.
3.  **Train**: Configure `training/lpg_training/train.py` and run the script:
    ```bash
    python -m training.lpg_training.train
    ```

## Roadmap

-   [ ] **Release of pre-trained models** for both LPG and PREVAL.
-   [ ] **Comprehensive documentation** and API references.
-   [ ] **Example Jupyter notebooks** for easier use and exploration.
-   [ ] **Full end-to-end inference script** for generating a presentation from a document.
-   [ ] Support for more design styles and customization.

## Contributing

We welcome community feedback and contributions! If you encounter any issues, have suggestions for improvement, or find a bug, please don't hesitate to **open an issue** on our GitHub tracker. Your feedback is valuable for refining this work.

## Citation

If you use RCPS in your research, please cite our paper:

```bibtex
@misc{your_name_2025_rcps,
  title={Multi-Agent Synergy-Driven Iterative Visual Narrative Synthesis},
  author={Your Name and Collaborator Name and ...},
  year={2025},
  eprint={YOUR_ARXIV_ID},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
