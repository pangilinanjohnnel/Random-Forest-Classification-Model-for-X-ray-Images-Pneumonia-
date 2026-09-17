# Classification of Chest X-Ray Images (Pneumonia)

A machine learning project for classifying pediatric chest X-ray images as **Pneumonia** or **Normal** using a Random Forest Classifier.

> **Note:** This project is for educational and research purposes only. The model is not intended to diagnose patients or replace professional clinical judgment.

---

## Overview

Pneumonia remains an important cause of childhood morbidity and mortality. According to UNICEF, in its article *"A child dies of pneumonia every 43 seconds"* published in November 2025, pneumonia kills more than 700,000 children under five every year, including approximately 190,000 newborns.

As a former Radiologic Technologist who worked in a public hospital, I encountered pneumonia frequently during chest X-ray examinations. These included community-acquired pneumonia as well as infections acquired within healthcare settings. Pneumonia was also frequently encountered as a severe complication associated with COVID-19.

This project explores whether a machine learning model can classify pediatric chest X-ray images into **Pneumonia** and **Normal** categories.

The long-term concept is an image-classification system that could potentially serve as an **assistive tool within a digital radiology workflow**, helping identify examinations that may require further clinical assessment.

---

## Problem

Pneumonia is a common respiratory infection, and chest X-ray is commonly used as part of its clinical assessment.

An automated image-classification system could potentially assist healthcare workflows by identifying chest X-rays showing patterns associated with Pneumonia and helping prioritize cases for further evaluation.

## Objective

Develop a classification model that distinguishes between:

* **Pneumonia**
* **Normal**

using pediatric chest X-ray images.

---

## Dataset

**Dataset:** Chest X-Ray Images (Pneumonia)
**Source:** Kaggle
**Categories:** Pneumonia and Normal
**Images:** 5,863 JPEG images

**Dataset:**
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

The dataset is organized into three folders:

```text
chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
```

The images consist of anterior-posterior chest X-rays from pediatric patients between one and five years old from Guangzhou Women and Children's Medical Center in Guangzhou, China.

The radiographs were obtained as part of routine clinical care. Low-quality and unreadable scans were removed during quality control. Diagnoses were evaluated by two expert physicians, while the evaluation set was additionally reviewed by a third expert to help account for potential grading errors.

---

## Methodology

### 1. Load the Images

The training and testing images were loaded from their respective folders.

Each image was:

* Converted to grayscale
* Resized to **64 × 64 pixels**
* Converted into a NumPy array
* Flattened into a one-dimensional feature vector

The target variable was the corresponding image category:

```text
NORMAL
PNEUMONIA
```

### 2. Prepare the Data

Separate predictor and target variables were created:

```text
X_train
y_train
X_test
y_test
```

### 3. Train the Model

A **Random Forest Classifier** was trained using the training dataset.

```python
RandomForestClassifier(random_state=42)
```

`random_state=42` was used to improve reproducibility.

Random Forest is an ensemble classification algorithm that combines predictions from multiple decision trees.

### 4. Evaluate the Model

The model was evaluated using:

* Confusion Matrix
* Classification Report
* Accuracy
* Sensitivity
* Specificity
* ROC-AUC
* ROC Curve

---

## Model

### Random Forest Classifier

Random Forest combines multiple decision trees to produce a classification result.

In this project, the chest X-ray images were used as input features, while the target classes were **Pneumonia** and **Normal**.

The model was configured with:

```python
RandomForestClassifier(random_state=42)
```

No hyperparameter tuning was performed in the current implementation.

---

## Results

The model produced approximately the following results:

| Metric      |     Result |
| ----------- | ---------: |
| Accuracy    | ~0.78–0.79 |
| Sensitivity |      ~0.99 |
| Specificity | ~0.41–0.45 |
| ROC-AUC     |      ~0.94 |

### Interpretation

**Accuracy (~0.78–0.79)**

Accuracy was relatively stable across runs. However, accuracy alone does not fully describe model performance because the dataset contains substantially more Pneumonia images than Normal images.

**Sensitivity (~0.99)**

The model achieved very high sensitivity, meaning it identified almost all Pneumonia cases in the test data.

**Specificity (~0.41–0.45)**

Specificity was considerably lower. This means the model had difficulty correctly identifying Normal cases and classified a substantial number of Normal images as Pneumonia.

**ROC-AUC (~0.94)**

The ROC-AUC indicates strong class-separation ability across different classification thresholds.

Overall, the results show a difference between the model's ability to separate the two classes and its performance at the default **0.5 classification threshold**.

---

## ROC Curve

The ROC curve was generated to examine the model's ability to distinguish between Pneumonia and Normal images across different classification thresholds.

![ROC Curve](images/roc_curve.png)

---

## Conclusion

The Random Forest model demonstrated strong class-separation performance, reflected by an ROC-AUC of approximately **0.94** and sensitivity of approximately **0.99**.

However, the relatively low specificity indicates that the model frequently classified Normal images as Pneumonia at the default classification threshold.

This demonstrates why classification performance should not be evaluated using accuracy alone. Sensitivity, specificity, and ROC-AUC provide additional information about how the model behaves.

The model therefore demonstrates potential as an **experimental classification approach**, but further development and validation would be required before considering any application in an actual clinical workflow.

---

## Scope and Limitations

* The dataset is limited to pediatric patients between approximately one and five years old.
* The classification task is limited to two classes: Pneumonia and Normal.
* The dataset contains substantially more Pneumonia images than Normal images, resulting in class imbalance.
* The dataset originates from a specific clinical population and institution, which may limit generalization to other hospitals, populations, imaging equipment, and clinical settings.
* Images were converted to grayscale and resized to 64 × 64 pixels, which may result in loss of visual information.
* Random Forest may not be the most suitable approach for extracting complex spatial features from medical images.
* No hyperparameter tuning was performed in the current implementation.
* The validation set was not used because it contained only 16 images, which was considered insufficient for reliable validation.
* The current implementation therefore uses the training set for model fitting and the test set for evaluation.
* No hyperparameter tuning or threshold optimization was performed using a validation set.

---

## Future Improvements

Potential future improvements include:

* Hyperparameter tuning
* Using a larger and more balanced dataset
* Evaluating additional classification models
* Exploring Convolutional Neural Networks (CNNs)
* Exploring transfer learning using pretrained image models
* Investigating classification threshold optimization
* Using a larger validation dataset
* Evaluating model performance on external datasets
* Investigating explainability methods for medical image classification

---

## Ethical Considerations

This model should be considered an **assistive tool rather than a replacement for clinical judgment**.

A positive model prediction should not be treated as a definitive diagnosis of Pneumonia. Final interpretation and diagnosis should remain the responsibility of qualified healthcare professionals, particularly radiologists and physicians who can consider the patient's clinical history, physical examination, laboratory findings, and other relevant information.

This consideration is particularly important for pediatric patients, where incorrect classification could potentially affect clinical decision-making.

Any future clinical implementation would require appropriate validation, monitoring, human oversight, patient safety considerations, and protection of patient privacy.

---

## Technologies Used

* Python
* NumPy
* scikit-learn
* Matplotlib
* Pillow (PIL)
* Random Forest

---

## Project Structure

```text
pediatric-pneumonia-classification/
│
├── README.md
├── notebook/
│   └── pneumonia_classification.ipynb
│
├── images/
│   └── roc_curve.png
│
└── requirements.txt
```

---

## Author

**Johnnel V. Pangilinan**

Former Radiologic Technologist | Data Science Graduate Student

This project combines medical imaging experience with machine learning to explore applications of data science in healthcare.

---

## Disclaimer

This project is intended for **educational and research purposes only**.

It is not a medical device, diagnostic system, or substitute for professional medical evaluation. The model has not been clinically validated and should not be used to make patient-care decisions.
