# Multimodal Attack Detection for Action Recognition Models


This is the official repository for VLAD (Vision-Language Attack Detection) as proposed in the paper "Multimodal Attack Detection for Action Recognition Models." VLAD is an adversarial attack detection method for video action recognition models.

VLAD is compatible with any action recognition model, and any implementation of CLIP can be used. In this official implementation, we use CSN from the PyTorchVideo library as the target action recognition model, and CLIP from the Huggingface Transformers package.

In this official implementation, we demonstrate the score calculation proposed in VLAD. Scores are calculated for both clean videos and attacked videos. We use PGD-v as the adversarial attack. The expected behavior is that clean videos will have low scores, while attacked videos will have high scores. More details can be found in our [paper](https://arxiv.org/pdf/2404.10790).

## Installation

To install the required packages:

```
pip install -r requirements.txt
```

## Data

Here we provide 8 example videos from Kinetics dataset where they can be directly used in our scripts.

Download link will be added soon.

## Usage


## Cite

```
@article{mumcu2024multimodal,
  title={Multimodal Attack Detection for Action Recognition Models},
  author={Mumcu, Furkan and Yilmaz, Yasin},
  journal={arXiv preprint arXiv:2404.10790},
  year={2024}
}
```
