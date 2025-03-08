<p align="center">
  <h2 align="center" style="margin-top: -30px;">MLLMs Know Where to Look: <br>Training-free Perception of Small Visual Details with Multimodal LLMs</h2>
</p>

<div style="font-family: charter;" align="center">
    <a href="https://saccharomycetes.github.io/" target="_blank">Jiarui Zhang</a>,
    <a href="https://mahyarkoy.github.io/" target="_blank">Mahyar Khayatkhoei</a>,
    <a href="https://www.prateekchhikara.com/" target="_blank">Prateek Chhikara</a>,
    <a href="https://www.ilievski.info/" target="_blank">Filip Ilievski</a>
</div>

<br>

<p align="center">
  <img src="images/method_case.png" alt="Method Overview" width="600">
</p>

[![arXiv](https://img.shields.io/badge/arXiv-2502.17422-b31b1b.svg)](https://arxiv.org/abs/2502.17422)
[![OpenReview](https://img.shields.io/badge/OpenReview-DgaY5mDdmT-blue.svg)](https://openreview.net/forum?id=DgaY5mDdmT)
[![ICLR 2025](https://img.shields.io/badge/ICLR-2025-green.svg)](https://iclr.cc/Conferences/2025)

## 📋 Overview

This repository contains the official implementation of our ICLR 2025 paper "MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs". Our method enables multimodal large language models (MLLMs) to better perceive small visual details without any additional training. This repository provides the detailed implementation of applying our methods on multiple MLLMs and benchmark datasets. 

## 🛠️ Installation

### Setup Environment
```bash
# Create and activate conda environment
conda create -n mllms_know python=3.10
conda activate mllms_know

# Install dependencies
pip install -r requirements.txt

# Install modified transformers library
cd transformers
pip install -e .
cd ..
```

## 🚀 Quick Start

We provide a [quick start notebook](quick_start.ipynb) that demonstrates how to:
- Load and process images
- Apply our methods to enhance visual perception
- Visualize attention maps

## 📊 Benchmark Evaluation

### Dataset Preparation
1. Download the benchmark datasets and corresponding images to your local directory
2. Update the paths in `info.py` with your local directory paths

Example (textvqa)

Dataset preparation:
```bash
mkdir -p data/textvqa/images
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P data/textvqa/images
unzip data/textvqa/images/train_val_images.zip -d data/textvqa/images
rm data/textvqa/images/train_val_images.zip
mv data/textvqa/images/train_images/* data/textvqa/images
rm -r data/textvqa/images/train_images
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P data/textvqa
```

Dataset processing (to a unified format):
```python
import json

with open('data/textvqa/TextVQA_0.5.1_val.json') as f:
    datas = json.load(f)

new_datas = []
for data_id, data in enumerate(datas['data']):
    data_id = str(data_id).zfill(10)
    question = data['question']
    labels = data['answers']
    image_path = f"{data['image_id']}.jpg"
    new_data = {
        'id': data_id,
        'question': question,
        'labels': labels,
        'image_path': image_path
    }
    new_datas.append(new_data)

with open('data/textvqa/data.json', 'w') as f:
    json.dump(new_datas, f, indent=4)
```


### Running Evaluations
To run our method on benchmark datasets, use the provided script:

```bash
# Format: bash run_all.sh [dataset] [model] [method]
bash run_all.sh textvqa llava rel_att
```

Get the model's performance:
```bash
python get_score.py --data_dir ./data/results --save_path ./
```

### Supported Datasets
- TextVQA
- DocVQA
- GQA
- AOKVQA
- POPE
- VSTAR
- VQAv2

### Supported Models
- LLaVA-1.5 (`llava`)
- InstructBLIP (`blip`)

For implementation details, see `llava_methods.py` and `blip_methods.py`. Please feel free to explore other MLLMs!

## 📝 Method Details

Our approach leverages the inherent attention mechanisms and gradients within MLLMs to identify regions of interest in images without requiring additional training. The key methods include:

1. **Relative Attention**: Compares attention patterns between task-specific and general prompts
2. **Gradient Attention**: Uses gradient information to identify important image regions
3. **High-Resolution Processing**: Enhances detail perception through multi-scale processing

For implementation details, see `llava_methods.py` and `blip_methods.py`.

## 📊 Results

Our method significantly improves MLLMs' performance on tasks requiring perception of small visual details, such as text recognition in images, fine-grained object recognition, and spatial reasoning. Please refer to the [paper](https://arxiv.org/abs/2502.17422) for more details and run the demo notebook for better understanding!

## 📚 Citation

If you find our paper and code useful for your research and applications, please cite using this BibTeX:
```bibtex 
@article{zhang2025mllms,
  title={MLLMs Know Where to Look: Training-free Perception of Small Visual Details with Multimodal LLMs},
  author={Zhang, Jiarui and Khayatkhoei, Mahyar and Chhikara, Prateek and Ilievski, Filip},
  journal={arXiv preprint arXiv:2502.17422},
  year={2025}
}

@inproceedings{zhang2025mllms,
  title={{MLLM}s Know Where to Look: Training-free Perception of Small Visual Details with Multimodal {LLM}s},
  author={Jiarui Zhang and Mahyar Khayatkhoei and Prateek Chhikara and Filip Ilievski},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=DgaY5mDdmT}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
