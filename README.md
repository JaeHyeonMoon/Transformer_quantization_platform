트랜스포머 경량화를 위한 차세대 양자화 기술 개발 및 오픈소스 벤치마크 구축
---

### About

This is the open-source transformer quantization platform for the project, titled "Developing quantization techniques and benchmark for efficient transformer".

### Requirements

- torch >= 1.8.1
- torchvision >= 0.9.1
- python >= 3.8.8
- timm == 0.6.13

### Datasets

- ImageNet (ILSVRC-2012) available at http://www.image-net.org

### Model Pool
- ## Supporting Transformer Model types
- ### ViT: Tiny/Small/Base
- Set args "--name={vit_tiny, vit_small, vit_base}"
- ### DeiT: Tiny/Small/Base
- Set args "--name={deit_tiny, deit_small, deit_base}"
- ### Swin: Tiny/Small/Base
- Set args "--name={swin_tiny, swin_small, swin_base}"

---
### Code
- 4-bit quantization & inference\
`python main.py --channel_wise=True --head_stem_8bit=True --name=(model name) --test_before_calibration --eps=1e-4 --n_bits=4 --n_groups=8`

- 6-bit quantization & inference\
`python main.py --channel_wise=True --head_stem_8bit=True --name=(model name) --test_before_calibration --eps=1e-4 --n_bits=6 --n_groups=8`
----

### Accuracy of ImageNet

| Method | #bits | ViT-T | ViT-S | ViT-B | DeiT-T | DeiT-S | DeiT-B | Swin-S | Swin-B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full-precision | 32 | 75.47 | 81.39 | 84.54 | 72.21 | 79.85 | 81.80 | 83.23 | 85.27 |
| PTQ4ViT <br>[ECCV '22] | 4 | 17.45 | 42.57 | 30.69 | 36.96 | 34.08 | 64.39 | 76.09 | 74.02 |
| RepQ-ViT <br>[ICCV '23] | 4 | - | 65.05 | 68.48 | 57.43 | 69.03 | 75.61 | 79.45 | 78.32 |
| IGQ-ViT <br>[CVPR '24] | 4 | 55.29 | 73.18 | 78.28 | 62.25 | 74.23 | 79.04 | 80.66 | 83.02 |
| I&S-ViT <br>[TPAMI '25] | 4 | - | 74.87 | 80.07 | 65.21 | 75.81 | 79.97 | 81.17 | 82.60 |
| DopQ-ViT <br>[arxiv '25] | 4 | - | 75.69 | **80.95** | 65.54 | 75.84 | 80.13 | 81.71 | 83.34 |
| **Ours** | 4 | **65.69** | **75.74** | 76.93 | **66.50** | **76.46** | **80.54** | **81.99** | **83.71** |
| PTQ4ViT <br>[ECCV '22] | 6 | 64.46 | 78.63 | 79.57 | 68.21 | 76.47 | 79.29 | 81.49 | 83.22 |
| RepQ-ViT <br>[ICCV '23] | 6 | - | 80.43 | 83.62 | 70.76 | 78.90 | 81.27 | 82.79 | 84.57 |
| IGQ-ViT <br>[CVPR '24] | 6 | 73.19 | 80.48 | 83.46 | 70.92 | 79.04 | 81.44 | 82.65 | 84.62 |
| I&S-ViT <br>[TPAMI '25] | 6 | - | 80.43 | 83.82 | 70.85 | 79.15 | 81.68 | 82.89 | 84.94 |
| DopQ-ViT <br>[arxiv '25] | 6 | - | 80.52 | **84.02** | 71.17 | 79.30 | 81.69 | 82.95 | 84.97 |
| **Ours** | 6 | **74.73** | **81.10** | 84.01 | **71.69** | **79.48** | **81.89** | **82.95** | **85.01** |
