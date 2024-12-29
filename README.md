# FISSP Dataset

**Fine-Grained Images for Self-Supervision Pre-training (FISSP)** is a diverse, high-quality dataset designed to enhance self-supervised pre-training for fine-grained image retrieval (FGIR). This repository contains:

- The FISSP dataset organization (train, validation, and test sets).
- Annotation files, including bounding boxes and label information.
- Scripts for dataset setup, preprocessing, and evaluation.
- Example models and configurations for self-supervised training and downstream FGIR tasks.

## Table of Contents

- [FISSP Dataset](#fissp-dataset)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features \& Statistics](#features--statistics)
  - [Installation \& Requirements](#installation--requirements)
  - [Usage](#usage)
  - [Citing FISSP](#citing-fissp)
  - [License](#license)

---

## Introduction

Fine-Grained Image Retrieval (FGIR) is a crucial task in the field of Computer Vision, requiring models to discriminate among subtle details in images. **FISSP** aims to mitigate the lack of fine-grained data in existing self-supervised training corpora by providing a large-scale set of images with high-quality annotations.

Our paper, ***"FISSP: A Novel Fine-Grained Image Dataset for Enhancing Image Retrieval through Self-Supervised Pre-Training"***, introduces FISSP and demonstrates its effectiveness on multiple benchmarks.

---

## Features & Statistics

Some highlights of FISSP:
- **Number of images**: 12M+ across fine-grained categories
- **Numerous Categories**: categories such as animals, plants, fungi, products, food, means of transportation, clothing, and other items relevant to fine-grained retrieval applications.

For detailed statistics, see [`docs/FISSP_Statistics.md`](docs/FISSP_Statistics.md).

---

## Installation & Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/qingxiaoli/FISSP.git
   cd FISSP
   ```
2.	Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. (Optional) Create a virtual environment:
   ```bash
   conda create -n fissp_env python=3.9
    conda activate fissp_env
    pip install -r requirements.txt
    ```

## Dataset Structure
The repository provides the following directory structure for data/:
```bash
    data/
    ├── train/
    ├── val/
    └── test/
```

## Usage
1.	Download/Set up the dataset
    Run the setup script:
    ```bash
    python scripts/download_fissp.py --data_dir ./data
    ```
    - This will download the FISSP images and extract them into ./data/train, ./data/val, and ./data/test.
	- Alternatively, if you already have the dataset, simply point the --data_dir to its location.

2. Preprocess the dataset
To resize, normalize, or reorganize data:
    ```bash
    python scripts/prepare_dataset.py --data_dir ./data
    ```
    Adjust parameters (e.g., image size) inside the script or via command-line arguments.


## Citing FISSP

If you find our dataset **FISSP** helpful for your research or project, please cite our manuscript as follows:

```bibtex
@unpublished{XiaoqingLi2024fissp,
    title  = {{FISSP}: A Novel Fine-Grained Image Dataset for Enhancing Image Retrieval through Self-Supervised Pre-Training},
    author = {Xiaoqing Li, Tao Hong & Ya Wang},
    note   = {Manuscript under submission, awaiting publication},
    year   = {2024}
}
```

## License

This project (FISSP Dataset) is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

Make sure to check the license terms if you plan to use the dataset. **Commercial or for-profit usage is strictly prohibited under this license.**
