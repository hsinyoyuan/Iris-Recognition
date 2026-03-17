# Iris Recognition

This is an iris recognition pipeline in Python. This project implements classical computer vision techniques for iris segmentation, normalization, feature encoding, and template matching.

## Pipeline Overview

The iris recognition system consists of four main stages:

1. **Iris Segmentation**
   - Detect iris and pupil boundaries
   - Remove eyelid occlusion

2. **Normalization**
   - Apply Daugman's rubber-sheet model
   - Convert iris region into polar coordinates

3. **Feature Extraction**
   - Extract iris texture features
   - Encode iris patterns into binary templates

4. **Matching**
   - Compare iris templates using Hamming distance

## Installation

Clone the repository:

```bash
git clone https://github.com/hsinyoyuan/Iris-Recognition.git
cd Iris-Recognition
```

Install the required packages. It is recommended to create a separate Python environment before installation.

- Python 3.10 (tested with Python 3.10.20)

```bash
pip install -r requirements.txt
```

## Usage

### Run the pipeline

```bash
python -m scripts.run \
--input input_list/list_Ganzin-J7EF-Gaze.txt \
--output result/result.txt
```

This performs:

- iris segmentation  
- normalization  
- feature extraction and encoding  
- template matching  

### Run evaluation

```bash
python scripts/eval.py --input result/result.txt
```

The evaluation script computes matching results and separates:

- **genuine pairs**
- **imposter pairs**

### Example workflow

```bash
python -m scripts.run \
--input input_list/list_Ganzin-J7EF-Gaze.txt \
--output result/result.txt

python scripts/eval.py --input result/result.txt
```

## Data Format

### Dataset Structure

The dataset is organized as follows:

```text
dataset/<dataset_name>/<subject_id>/<eye>/<image_name>
```

Example:

```text
dataset/CASIA-Iris-Lamp/001/L/S2001L06.jpg
dataset/CASIA-Iris-Lamp/075/R/S2075R09.jpg
```

Where:

- `dataset_name` represents the dataset to be used
- `subject_id` represents the identity
- `L` / `R` indicates the left or right eye
- `image_name` is the iris image file

### Input Format

The system reads an input `.txt` file where each line contains a pair of image paths to be compared.

Example:

```text
dataset/CASIA-Iris-Lamp/001/L/S2001L06.jpg dataset/CASIA-Iris-Lamp/001/L/S2001L07.jpg
dataset/CASIA-Iris-Lamp/001/L/S2001L06.jpg dataset/CASIA-Iris-Lamp/075/R/S2075R09.jpg
```

## Project Structure

```text
iris_recognition/
├── dataset/
│   ├── CASIA-Iris-Lamp/
│   ├── CASIA-Iris-Thousand/
│   └── Ganzin-J7EF-Gaze/
├── input_list/
│   ├── list_CASIA-Iris-Lamp.txt
│   ├── list_CASIA-Iris-Thousand.txt
│   └── list_Ganzin-J7EF-Gaze.txt
├── scripts/
│   ├── run.py
│   └── eval.py
├── src/
│   ├── __init__.py
│   ├── segmentation.py
│   ├── normalization.py
│   ├── feature_extraction.py
│   ├── encoding.py
│   ├── matching.py
│   └── processor.py
├── README.md
├── requirements.txt
├── LICENSE
└── reference repo.txt
```

### Directory Description

- `dataset/`  
  Dataset directory. In this project, example dataset structure is shown for reference only.

- `input_list/`  
  Text files containing image pairs or input samples for different datasets.

- `scripts/`  
  Main scripts for running the pipeline and evaluation.
  - `run.py`: execute the iris recognition pipeline
  - `eval.py`: evaluate matching performance

- `src/`  
  Core implementation of the iris recognition pipeline.
  - `segmentation.py`: iris and pupil localization
  - `normalization.py`: rubber-sheet normalization
  - `feature_extraction.py`: iris texture extraction
  - `encoding.py`: binary template encoding
  - `matching.py`: template comparison with Hamming distance
  - `processor.py`: pipeline integration / processing flow

## Notes on Data Availability

Due to file size limitations and dataset usage restrictions, the full datasets are not included in this repository.

## Acknowledgments

This project was developed with reference to the following open-source implementation:

- [mvjq/IrisRecognition](https://github.com/mvjq/IrisRecognition)  
  A classical iris recognition project that includes iris localization, normalization, feature encoding, and matching.