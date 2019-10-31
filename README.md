# pytorch-template (v0.2.0-alpha)

---

pytorchプロジェクトを簡単に開発するためのテンプレート。

## Description

## Demo

## Requirement
- torch, torchvision
- tensorboardX
- tqdm

## Usage
```python
pytorch_template/
├── run/
├── dataloader/
│   ├── __init__.py
│   ├── custom_transforms.py
│   ├── dataset.py        <--- "change this file!"
│   └── utils.py
├── modeling/        <--- "change this module!"
│   ├── modeling.py
│   └── sub_module.py
├── sample_data/
│   ├── <image_index>.png
│   └── label.csv
├── utils/
│   ├── metrics.py
│   ├── saver.py
│   └── summaries.py
├── working_top.ipynb
├── __init__.py
├── config.py        <--- "change this file!"
├── predict.py
├── train.py        <--- "change this file!"
└── README.md
```


## Install

## Contribution

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[Shotaro Kataoka](https://github.com/ShotaroKataoka)
