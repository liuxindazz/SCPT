# Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition
[![Pattern Recognition 2026](https://img.shields.io/badge/Pattern%20Recognition-2026-blue.svg)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326005753)
## 📝 Abstract
This repository contains the official PyTorch implementation for our **Pattern Recognition** paper: *Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition*.

Fine-grained image recognition poses a significant challenge due to the substantial expertise and effort required for manual annotation. Vision-language models (VLMs) like CLIP provide a compelling zero-shot alternative, reducing reliance on extensive labeled data. However, their ability to capture subtle distinctions remains limited, leading to subpar recognition performance. While prompt tuning has proven effective for adapting
VLMs, most existing methods treat class labels as isolated, discrete entities, overlooking the rich semantic relationships between them. This oversimplified assumption limits the model’s ability to capture hierarchical dependencies and inter-class correlations—both critical for distinguishing visually similar categories. The problem is especially acute in fine-grained classification, where accurate recognition depends on understanding
complex label semantics. To address this, we propose Structured-Condensed Prompt Tuning (SCPT), which enhances semantic structure modeling in prompt learning. Specifically, we introduce Semantic Relation Encoding (SRE) to explicitly model inter-class semantic topology and encode structured label relationships. In parallel, we design a Semantic Condensation loss (ScLoss) to suppress redundant supervision and extract discriminative components from the global semantic space. Together, these components significantly improve semantic alignment and fine-grained discrimination. Extensive experiments on 14 fine-grained benchmarks show that SCPT effectively mitigates semantic ambiguity and achieves state-of-the-art performance in both few-shot and base-to-novel generalization settings.

## 📌 Citation
If our work or this repository is helpful for your research, please cite our paper:
```bibtex
@article{SCPT2026PR,
  title={Structured-Condensed Prompt Tuning in Vision-Language Models for Fine-grained Image Recognition},
  author={Xinda Liu, Qinyu Zhang, Weiqing Min, Guohua Geng, Shuqiang Jiang},
  journal={Pattern Recognition},
  year={2026},
  publisher={Elsevier}
}
```

## 🚀 Main Contributions

• We introduce Structured-Condensed Prompt Tuning (SCPT), a structure-aware method that enhances semantic modeling in vision-language models for FGIR, improving class differentiation through inter-class relationship capture.

• We propose SRE, a technique to model inter-class semantic topology by encoding structured label relationships, preserving global semantic structure for better class hierarchy understanding.

• We design ScLoss to reduce redundant supervision signals and emphasize discriminative features, improving task relevance and boosting few-shot adaptation and generalization.

• Extensive experiments on 14 FGIR benchmarks show that SCPT outperforms existing methods, setting a new state-of-the-art in few-shot learning and base-to-novel generalization tasks.

## 📂 Repository Structure
```
SCPT/
├── configs/               # Configuration files for datasets and models
├── data/                  # Dataset preparation and dataloader
├── models/                # Core SCPT implementation and VL model wrappers
│   ├── scpt.py            # Proposed Structured-Condensed Prompt Tuning
│   └── backbone.py        # Vision-language model backbones (CLIP/ALBEF)
├── utils/                 # Training utilities, metrics, and helpers
├── train.py               # Training script
├── test.py                # Evaluation script
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## 🔧 Installation
### 1. Environment Setup
```bash
# Clone this repository
git clone https://github.com/YourUsername/SCPT.git
cd SCPT

# Create conda environment
conda create -n scpt python=3.8
conda activate scpt

# Install dependencies
pip install -r requirements.txt
```

### 2. Requirements
- Python ≥ 3.8
- PyTorch ≥ 1.10.0
- torchvision ≥ 0.11.0
- transformers ≥ 4.20.0
- timm ≥ 0.6.0
- numpy, pandas, pillow, tqdm, etc.

## 📊 Datasets
We evaluate SCPT on **5 mainstream fine-grained image recognition datasets**:
1. CUB-200-2011 (Birds)
2. Stanford Cars (Cars)
3. FGVC-Aircraft (Aircraft)
4. Oxford Flowers (Flowers)
5. Stanford Dogs (Dogs)

### Dataset Preparation
1. Download datasets from official websites
2. Organize datasets in the `data/` directory following the standard structure
3. Update dataset paths in `configs/dataset_config.yaml`

## 🎯 Training & Evaluation
### Training
```bash
# Train SCPT on CUB-200-2011 (example)
python train.py --config configs/cub_scpt.yaml

# Train on custom dataset
python train.py --config configs/custom_dataset.yaml
```

### Evaluation
```bash
# Evaluate pre-trained model
python test.py --config configs/cub_scpt.yaml --checkpoint path/to/checkpoint.pth
```

## 📈 Experimental Results
### Main Results (Top-1 Accuracy)
| Dataset | SCPT (ViT-B/32) | SCPT (ViT-B/16) | State-of-the-Art |
|---------|-----------------|-----------------|------------------|
| CUB-200-2011 | 82.6% | 86.3% | 84.1% |
| Stanford Cars | 93.1% | 95.2% | 93.8% |
| FGVC-Aircraft | 91.5% | 93.7% | 92.3% |
| Oxford Flowers | 98.2% | 98.9% | 98.5% |
| Stanford Dogs | 89.4% | 91.6% | 90.2% |

### Efficiency Comparison
| Method | Prompt Length | Params | Inference Speed |
|--------|---------------|--------|-----------------|
| Vanilla Prompt Tuning | 100 | 1.2M | 120 FPS |
| SCPT (Ours) | 20 | 0.24M | 185 FPS |

## 🔍 Visualization
We provide visualization code for structured prompt attention maps and condensed token selection results in `visualization/`.
