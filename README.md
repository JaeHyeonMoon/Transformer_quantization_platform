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
- ViT: Tiny/Small/Base
- DeiT: Tiny/Small/Base
- Swin: Tiny/Small/Base
---
### Code
- 4-bit quantization & inference\
`python main.py --channel_wise=True --head_stem_8bit=True --name=(model name) --test_before_calibration --eps=1e-4 --n_bits=4 --n_groups=8`

- 6-bit quantization & inference\
`python main.py --channel_wise=True --head_stem_8bit=True --name=(model name) --test_before_calibration --eps=1e-4 --n_bits=6 --n_groups=8`
