# Hyperbolic PCA and MDS Experiment Runner

This project provides a script to run experiments with different hyperbolic PCA (Principal Component Analysis) and MDS (Multidimensional Scaling) models on various datasets. It supports multiple methods such as SFPCA, Horopca, BSA, and PGA.

## Prerequisites

- Python 3.x
- Required Python packages listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/puoya/space_form_pca.git
    cd yourrepository
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Command-Line Arguments

The script `run.py` accepts the following command-line arguments:

- `--dataset`: Name of the dataset (default: `unfiltered`).
- `--model`: Model to run (`sfpca`, `horopca`, `bsa`, `pga`; default: `sfpca`).
- `--dimension`: Dimension parameter for the model (default: `10`).

### Running the Script

To run the script with default values:
```bash
python3 run.py
