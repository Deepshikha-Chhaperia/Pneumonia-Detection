# Pneumonia-Detection

## Dataset Overview:
The "Pneumonia Detection Using Chest X-Ray Images" project aims to develop a robust and accurate model for the detection of pneumonia from chest X-ray images in pediatric patients aged one to five years old. The dataset comprises 5,863 X-Ray images categorized into Pneumonia and Normal classes. These images were sourced from the Guangzhou Women and Children’s Medical Center as part of routine clinical care. The dataset is split into three folders: train, test, and val.

Dataset Source:
https://data.mendeley.com/datasets/rscbjbr9sj/2

## Dataset Description:
* Categories: Pneumonia and Normal
* Image Types: JPEG format
* Imaging Method: Anterior-posterior chest X-rays
* Quality Control: All images underwent initial quality screening to remove poor-quality scans. Diagnoses were assessed by two expert physicians before being used for AI training.
* Evaluation: A third expert reviewed the evaluation set to ensure grading accuracy.

## Dataset Folder Structure:
* Train Directory: Contains training images for model training.
* Test Directory: Contains images for model testing.
* Val Directory: Includes images for validation during model training.

## Model Development:
* Model Types: Convolutional Neural Networks (CNN) and Transfer Learning (Inception V3).
* Evaluation Metrics: Accuracy, F1 Score, Precision, Recall.
* Preprocessing: Augmentation was applied to enhance model generalization.
* Model Training: Training was performed using batch-wise generators and class weights for imbalanced data.

## Results:
* Initial Models: Demonstrated high recall for Pneumonia detection.
* Transfer Learning: Inception V3 model showed improvement in recall, precision, and accuracy.
* Class Weights: Improved recall while reducing accuracy and precision.

## Objective
The primary objective of this project is to build a deep learning model that can accurately classify chest X-ray images as showing signs of pneumonia or being normal. The model will undergo training, testing, and validation to ensure its efficacy in accurately identifying pneumonia cases, with the ultimate goal of aiding in early and accurate diagnosis.

## Problem Statement
Pneumonia is a serious health concern in pediatric patients, and timely and accurate diagnosis is crucial for effective treatment. However, manually diagnosing pneumonia from chest X-ray images can be challenging and time-consuming for healthcare professionals. The development of an automated pneumonia detection system using deep learning can significantly aid in the quick and accurate diagnosis of pneumonia, potentially reducing the turnaround time for diagnosis and improving patient outcomes.

By leveraging machine learning techniques, the project aims to address the following key challenges:

* Developing a highly accurate and reliable model for automated pneumonia detection from chest X-ray images.
* Minimizing false negatives to ensure that patients with pneumonia are not overlooked during diagnosis.
* Creating a scalable and efficient solution that can be applied to real-world clinical settings to support healthcare professionals in accurate diagnosis.

The project seeks to contribute to the advancement of medical image analysis and automation, with the ultimate goal of positively impacting patient care and healthcare efficiency.
