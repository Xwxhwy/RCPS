# RCPS: Reflective Coherent Presentation Synthesis

[![Paper-Link-Coming-Soon](https://img.shields.io/badge/Paper-Link-red?style=flat-square)](https://arxiv.org/abs/YOUR_PAPER_ID) <!-- Replace with your arXiv link -->
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

**Current Version: `v1.0.0`**

This initial release includes:
-   ✔️ The complete training pipeline for the Layout Prototype Generator (LPG).
-   ✔️ A functional implementation of the PREVAL evaluator.
-   ✔️ The core modules for LDL, layout interpretation, and rendering logic.

This codebase is under **active development**. We are committed to improving its structure, documentation, and functionality. Please see our roadmap below.

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

We are actively working on the following enhancements:

-   [ ] **Release of pre-trained models** for both LPG and PREVAL.
-   [ ] **Comprehensive documentation** and API references.
-   [ ] **Example Jupyter notebooks** for easier use and exploration.
-   [ ] **Full end-to-end inference script** for generating a presentation from a document.
-   [ ] Support for more design styles and customization.

## Contributing

We welcome community contributions! If you find a bug or have a suggestion, please open an issue.

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
