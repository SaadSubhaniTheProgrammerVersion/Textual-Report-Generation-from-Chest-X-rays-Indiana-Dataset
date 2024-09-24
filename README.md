# Textual Report Generation from Chest Radiographs using Deep Learning

Welcome to the **Textual Report Generation from Chest Radiographs** project! This repository contains all the necessary code to generate medical reports from chest X-rays using deep learning models. The project is based on the [Indiana Chest X-ray Dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university).

## Dataset

To get started, you'll need to download the dataset. The dataset includes a `reports.csv` file, which contains all the reports for the relevant chest X-rays. These reports are used to fine-tune the models for generating accurate textual descriptions.

## Project Structure

### 1. **Fine-tuning BioGPT**

- **File**: `BioGPT-FINAL.ipynb`
  
  This script fine-tunes the BioGPT model on the medical reports provided in the `reports.csv` file from the dataset. BioGPT, after fine-tuning, will generate high-quality medical reports based on chest X-ray embeddings.

### 2. **Setting Up ChexNet for Image Embeddings**

- **Folder**: `ChexNet-Report-Generation`
  
  Inside the `ChexNet-Report-Generation` folder, running `create_model.py` will download ChexNet's pre-trained weights and set it up to generate image embeddings from the chest X-ray images. These embeddings are crucial for feeding into the fine-tuned BioGPT model.

### 3. **Final Pipeline**

- **File**: `final.py`
  
  Once the ChexNet model is set up, make sure to update the correct path to the fine-tuned BioGPT model in `final.py`. This file contains the complete pipeline that combines image embeddings from ChexNet with the fine-tuned BioGPT to generate textual reports. You can also launch a simple frontend using **Streamlit**:
  
  ```bash
  streamlit run final.py
