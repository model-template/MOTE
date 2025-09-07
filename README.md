# A Responsible Face Recognition Approach for Small and Mid-Scale Systems Through Personalized Neural Networks

This repository contains the official PyTorch implementation for the paper:

**A Responsible Face Recognition Approach for Small and Mid-Scale Systems Through Personalized Neural Networks**
*Sebastian Groß, Stefan Heindorf, Philipp Terhörst*
*Paderborn University*

**[Link to the paper coming soon]**

## Introduction

This work introduces **MOTE (Model-Template)**, a novel face recognition approach that replaces traditional vector-based templates with small, personalized neural networks. This method enhances privacy, fairness, and explainability in face recognition systems, making it particularly suitable for small- and medium-scale applications where these factors are critical.

Our key contributions are:
- **Enhanced Privacy:** MOTE significantly reduces the risk of soft-biometric attribute inference (e.g., gender) from stored templates.
- **Improved Fairness:** Our approach allows for individual-level fairness adjustments during the enrollment process.
- **Explainability:** MOTE enables the use of standard explainability methods like Grad-CAM++ to visualize and understand the decision-making process of the models.

## Repository Structure

The refactored repository is organized as follows:

```
.
├── data/                     # Data files (.npy)
├── pretrained_models/        # To store trained MOTE models
├── results/                  # For storing evaluation results (JSON) and figures (PNG)
├── src/                      # All source code
│   ├── config.py             # Central configuration for paths and parameters
│   ├── data_loader.py        # Loads datasets and embeddings
│   ├── data_processor.py     # Helper functions for data manipulation
│   ├── kde_high_dimension.py # Core KDE logic for template generation
│   ├── mote.py               # MOTE classifier model definitions
│   ├── train.py              # Script for training MOTE models
│   ├── evaluate.py           # Script for performance, fairness, and privacy evaluation
│   └── visualize.py          # Script for generating plots and figures
├── README.md
├── requirements.txt
├── run_experiments.py        # Main script to run all experiments
└── LICENSE
```

## Setup

### Prerequisites

- Python 3.8+
- PyTorch
- Torchvision
- Numpy
- Scikit-learn
- Matplotlib
- Pandas
- Joblib
- Optuna

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/model-template/MOTE.git
    cd MOTE
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

### Datasets and Pre-trained Models

1.  **Download Datasets:** Download the Adience, LFW, and ColorFeret datasets.
2.  **Download Embeddings:** Download the pre-computed face embeddings for ArcFace and MagFace.
3.  **Organize Data:** Place the datasets and embeddings inside the `data/` directory. The expected structure is defined in `src/config.py`. Please modify the paths in `src/config.py` to match the location of your data.
4.  **Create Directories:** Create the `pretrained_models/` and `results/` directories in the root of the project if they don't exist.

## Reproducing the Paper's Results

The entire experimental pipeline can be run using the `run_experiments.py` script. This script will train all necessary models, run the evaluations, and save the results and figures to the `results/` directory.

### Running All Experiments

To run all experiments and generate all figures and tables from the paper, execute the following command:

```bash
python run_experiments.py --all
```

### Running Specific Experiments

You can also run specific parts of the evaluation pipeline.

**1. Train MOTE Models:**
Train all personalized MOTE models for a given dataset and face recognition architecture.

```bash
python run_experiments.py --train --architecture <arch> --dataset <dataset>
```
-   `<arch>`: `arcface` or `magface`
-   `<dataset>`: `adience`, `colorferet`, or `lfw`

**2. Evaluate Models:**
Run the performance, fairness, and privacy evaluations. This will save the raw results to a JSON file in the `results/` directory.

```bash
python run_experiments.py --evaluate --architecture <arch> --dataset <dataset>
```

**3. Generate Figures:**
Generate all paper figures from the saved evaluation results.

```bash
python run_experiments.py --visualize
```

**Example: Reproduce ArcFace results on ColorFeret**

```bash
# Step 1: Train all MOTE models for ArcFace on the ColorFeret dataset
python run_experiments.py --train --architecture arcface --dataset colorferet

# Step 2: Run the evaluation
python run_experiments.py --evaluate --architecture arcface --dataset colorferet

# Step 3: Generate plots (this will use the results from all evaluations)
python run_experiments.py --visualize
```

## Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{gross2025responsible,
  title={A Responsible Face Recognition Approach for Small and Mid-Scale Systems Through Personalized Neural Networks},
  author={Gro{\ss}, Sebastian and Heindorf, Stefan and Terh{"o}rst, Philipp},
  booktitle={Proceedings of the IEEE/CVF International Joint Conference on Biometrics (IJCB)},
  year={2025}
}
```
