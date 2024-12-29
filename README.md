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
  - [Evaluation](#evaluation)
  - [Citing](#citing)
  - [License](#license)
  - [Contact](#contact)
  - [Contributing](#contributing)

---

## Introduction

Fine-Grained Image Retrieval (FGIR) is a crucial task in the field of Computer Vision, requiring models to discriminate among subtle details in images. **FISSP** aims to mitigate the lack of fine-grained data in existing self-supervised training corpora by providing a large-scale set of images with high-quality annotations.

Our paper, ***"FISSP: A Novel Fine-Grained Image Dataset for Enhancing Image Retrieval through Self-Supervised Pre-Training"***, introduces FISSP and demonstrates its effectiveness on multiple benchmarks:
- [CUB-200-2011](https://www.vision.caltech.edu/datasets/)
- [Cars-196](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)
- [SOP (Stanford Online Products)](https://cvgl.stanford.edu/projects/lifted_struct/)
- [In-Shop Clothes Retrieval](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/InShopRetrieval.html)

---

## Features & Statistics

Some highlights of FISSP:
- **Number of images**: 200,000+ across fine-grained categories
- **Number of classes**: 500+ distinct classes
- **Annotations**:
  - Bounding boxes for each object
  - Fine-grained class labels
  - Train/Val/Test splits

For detailed statistics, see [`docs/FISSP_Statistics.md`](docs/FISSP_Statistics.md).

---

## Installation & Requirements

1. Clone this repository:
   ```bash
   git clone https://github.com/username/FISSP.git
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

Each folder is further divided into subdirectories by class, along with corresponding annotation files in annotations/. If you are downloading the dataset for the first time, see Usage.

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
3. Self-Supervised Pre-training
	- Example config: experiments/pretraining_config.yaml
	- Run:
        ```bash
        python experiments/run_experiment.py --config experiments/pretraining_config.yaml
        ```
    - This trains the self-supervised model (defined in models/self_supervised_model.py) using both contrastive and generative objectives.
4. Fine-Tuning for FGIR
	- Example config: experiments/finetuning_config.yaml
	- Run:
        ```bash
        python experiments/run_experiment.py --config experiments/finetuning_config.yaml
        ```
    - Loads the pre-trained weights and fine-tunes for retrieval tasks on standard benchmarks or custom splits.

## Evaluation

To evaluate on typical FGIR benchmarks, we provide scripts/evaluate_fgir.py. For example:
```bash
python scripts/evaluate_fgir.py \
    --model_path ./checkpoints/pretrained_model.pth \
    --data_dir ./data/test \
    --batch_size 32
```
This script computes metrics like Recall@1, Recall@5, and mAP (if appropriate).

## Citing
```bash
@inproceedings{FISSP2024,
  title     = {FISSP: A Novel Fine-Grained Image Dataset for Enhancing Image Retrieval through Self-Supervised Pre-Training},
  author    = {YourName and Co-Authors},
  booktitle = {Conference Name},
  year      = {2024},
  pages     = {1--10}
}
```

## License
This project is licensed under the Your Chosen License.
Make sure to check the license terms if you plan to use the dataset in commercial or derivative works.

## Contact
For questions regarding FISSP, please open a GitHub issue or contact us via:
> Email: your.email@domain.com
We welcome pull requests and collaboration from the community!

##  Contributing
Please see CONTRIBUTING.md for guidelines on how to contribute to this project.

---
Thank you for using FISSP! We hope our dataset will advance research in fine-grained image retrieval and self-supervised learning.
```bash

## Additional Notes

1. **Annotation Files**:  
   Include as much detail as possible in your annotation files (e.g., bounding boxes, part locations if relevant). JSON or CSV formats are common.

2. **Documentation**:  
   - `docs/FISSP_Statistics.md` can detail class distributions, sample images, or any special data collection procedures.  
   - `docs/FAQ.md` can help new users quickly resolve common issues.

3. **License**:  
   - Common choices for datasets include [Creative Commons licenses (e.g., CC-BY)](https://creativecommons.org/licenses/) if you want to allow broader usage but still require attribution.  
   - For code, you may choose [MIT](https://opensource.org/licenses/MIT), [Apache 2.0](https://opensource.org/licenses/Apache-2.0), or [GPL](https://opensource.org/licenses/GPL-3.0).

4. **Contributions**:  
   - Encourage the community to contribute scripts for new architectures, improved data augmentations, or different evaluation metrics.

5. **Version Control**:  
   - Consider tagging releases (e.g., `v1.0`, `v1.1`) if the dataset or code changes significantly.

With this structure, you’ll have a user-friendly, well-documented repository for **FISSP** that others can easily explore, download, and integrate into their own research pipelines. Good luck with your project!
```