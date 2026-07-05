# RF-DETR Person Detection

## Project Overview

This project implements the **RF-DETR (Receptive Field Detection Transformer)** object detection model using Python and PyTorch. The model was trained to detect a single object class (**person**) using a custom dataset converted into the COCO annotation format.

The project includes:

- Dataset conversion from CSV to COCO JSON
- Data preprocessing and augmentation
- RF-DETR model training
- Model evaluation
- Prediction visualization
- Intersection over Union (IoU) calculation

---

## Dataset

This project uses the **People Detection Dataset** created by **Adil Shamim** on Kaggle.

**Dataset:** : https://www.kaggle.com/datasets/adilshamim8/people-detection?resource=download

**Author:** Adil Shamim

**Description:**
The dataset contains annotated images for person detection in YOLO format, with predefined training, validation, and test splits. It was created using Roboflow and is intended for training object detection models such as RF-DETR, YOLO, and other deep learning architectures.

The dataset contains images of people divided into three subsets:

- Train (760imgs to 21imgs)
- Validation (15210imgs to 101imgs)
- Test (1431imgs to 21 imgs)

All are reduced due to high maintenance workload of cpu.

The original annotations were stored in CSV format and converted into COCO JSON format before training.

To speed up CPU training, a reduced dataset (`dataset_small`) was also created.

---

## Requirements

Python 3.10+

Required libraries:

```bash
pip install rfdetr
pip install "rfdetr[train,loggers]"
pip install albumentations
pip install supervision
pip install pandas
pip install torchvision
pip install matplotlib
pip install pillow
pip install opencv-python
```

---

## Project Structure

```
dataset/
    train/
    valid/
    test/

dataset_small/
    train/
    valid/
    test/

outputs/
    checkpoint_best_regular.pth

predictions/

main.ipynb
README.md
```

---

## Project Workflow

### 1. Dataset Preparation

- Convert CSV annotations into COCO JSON format.
- Verify that all image files exist.
- Generate a reduced dataset (`dataset_small`) for faster CPU training.

---

### 2. Data Preprocessing

Images are preprocessed using TorchVision transforms:

- Resize
- Random Horizontal Flip
- Random Rotation
- Color Jitter
- Tensor conversion
- ImageNet normalization

---

### 3. Model Implementation

The project uses **RF-DETR Base**, which includes:

- CNN Backbone
- Transformer Encoder
- Transformer Decoder
- Receptive Field Enhancement module
- Hungarian Matcher
- Feed Forward prediction heads

The pretrained RF-DETR model is fine-tuned on the custom dataset.

---

### 4. Model Training

Training configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 2 |
| Batch Size | 1 |
| Learning Rate | 0.0001 |
| Resolution | 336 |
| Optimizer | AdamW |
| Early Stopping | Enabled |
| Patience | 3 |
| Model Checkpoint | Enabled |

---

### 5. Model Evaluation

The trained model was evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- mAP@0.5
- mAP@0.5:0.95
- Average Recall
- Intersection over Union (IoU)

Prediction images with bounding boxes and confidence scores were also generated.

---

## Results

| Metric | Value |
|---------|------:|
| Accuracy | 53.10% |
| Precision | 54.24% |
| Recall | 52.03% |
| F1 Score | 53.11% |
| mAP@0.5 | 51.01% |
| mAP@0.5:0.95 | 28.59% |
| Average Recall | 45.93% |

---

## Visualization

Predicted bounding boxes are saved inside:

```
predictions/
```

Each image contains:

- Bounding box
- Predicted class
- Confidence score

---

## IoU Demonstration

The project includes a custom Python implementation of the **Intersection over Union (IoU)** metric to determine whether a prediction is considered a True Positive using the standard threshold of **IoU ≥ 0.5**.

---

## Notes

- Training was performed on CPU.
- A reduced dataset (`dataset_small`) was used to decrease training time.
- The pretrained RF-DETR weights were fine-tuned for a single class (**person**).

---

## Authors

- Leona Raine Aquino
- Bianca Lalaine Francisco
