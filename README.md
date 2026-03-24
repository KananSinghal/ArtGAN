[Jupyter](https://www.google.com/url?q=https://drive.google.com/file/d/1GuWEDUTcIQhq5fHw7CXdWQY__2x5BqBb/view?usp%3Dsharing&sa=D&source=editors&ust=1774365521454376&usg=AOvVaw2oDBZxdeO4R6joX6GUH7E3)




[PDF](https://www.google.com/url?q=https://drive.google.com/file/d/1BPCPAM4C6JXNH5ysAGkFXE0gBa_s-n7C/view?usp%3Dsharing&sa=D&source=editors&ust=1774365521454230&usg=AOvVaw01MywtpTQW9xus1Tygjpn8)


# ArtGAN

# Multi-Task Learning for WikiArt Classification
A deep learning project using PyTorch to simultaneously classify fine art paintings by **Style**, **Genre**, and **Artist**.

## Overview
This project implements a custom Multi-Task Learning (MTL) architecture. Unlike standard models that predict one attribute, this model shares a feature extractor to predict three distinct labels in a single forward pass.


## Architecture
The model uses a hybrid CNN-RNN approach to capture both local textures and global spatial relationships:
1. **Backbone:** EfficientNetV2-S (Pretrained on ImageNet).
2. **Sequential Processing:** The 6x6 spatial feature map is treated as a sequence of 36 tokens.
3. **RNN:** A 2-layer Bidirectional GRU processes the tokens to understand spatial context.
4. **Task Heads:** Three independent MLP heads for Style (16 classes), Genre (10 classes), and Artist (23 classes).

## Training Techniques
1. **Differential Learning Rates:** Backbone (1e-4) vs. Heads/GRU (3e-4).
2. **Class Imbalance Handling:** Inverse-frequency loss weights capped at 10.0 to handle rare classes like Pointillism.
3. **Augmentation:** Mixup augmentation to improve generalization.
4. **Optimization:** AdamW with Cosine Annealing learning rate decay.

## Dataset
The project uses the [WikiArt Dataset](https://www.kaggle.com/datasets/steubk/wikiart), 
[github](https://github.com/cs-chan/ArtGAN/blob/master/WikiArt%20Dataset/README.md)

The data pipeline includes:
  Automated filtering for missing image files.
  Unified CSV merging for multi-task label alignment.

## Usage
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Update the `CFG` dictionary in the notebook/script with your local data paths.
4. Run the training cell.
