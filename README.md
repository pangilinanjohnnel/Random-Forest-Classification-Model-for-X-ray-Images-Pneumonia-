# Classification of Chest X-Ray Images (Pneumonia)

A machine learning project for classifying pediatric chest X-ray images as **Pneumonia** or **Normal** using a Random Forest Classifier.

> **Note:** This project is for educational and research purposes only. The model is not intended to diagnose patients or replace professional clinical judgment.

---

## Overview

Pneumonia remains an important cause of childhood morbidity and mortality. According to UNICEF, in its article *"A child dies of pneumonia every 43 seconds"* published in November 2025, pneumonia kills more than 700,000 children under five every year, including approximately 190,000 newborns.

As a Radiologic Technologist who used to worked in a public hospital, I encountered pneumonia frequently during chest X-ray examinations. These included community-acquired pneumonia as well as infections acquired within healthcare settings also known as nosocomial infection. Pneumonia was also frequently encountered as a severe complication associated with COVID-19, a very common thing I see when I was active on duty during COVID-19 pandemic.

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

**Link:** https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Categories:** Pneumonia and Normal

**Number of images:** 5,863 JPEG images

The dataset is organized into three folders: train, test, and val, with images separated into the Pneumonia and Normal categories.

The images consist of anterior-posterior chest X-rays from pediatric patients between one and five years old from Guangzhou Women and Children’s Medical Center in Guangzhou, China. The radiographs were obtained as part of routine clinical care.

For quality control, low-quality and unreadable scans were removed. The diagnoses were evaluated by two expert physicians, while the evaluation set was additionally reviewed by a third expert to help account for potential grading errors.

## Methodology

1. The chest X-ray images were loaded from the training and testing datasets.
2. The images were converted into predictor data (X) while their corresponding categories, Pneumonia or Normal, were assigned as the target variable (y).
3. The training and testing data were organized into separate predictor and target sets:

   * `X_train`
   * `y_train`
   * `X_test`
   * `y_test`
4. A Random Forest Classifier was trained using the training data.
5. The model was configured with `random_state=42` to improve reproducibility.
6. Model performance was evaluated using a confusion matrix and classification report.
7. ROC-AUC was also calculated to evaluate the model's ability to distinguish between the two classes across different classification thresholds.
8. The ROC curve was plotted to visually examine the model's class-separation performance.

## Model

Model used was a **Random Forest Classifier** with `random_state=42`.

Random Forest is an ensemble classification algorithm that combines predictions from multiple decision trees. In this project, the chest X-ray images serve as the input data used by the classifier to learn patterns associated with the two target classes: Pneumonia and Normal. 

In concept if random forest based its decision on multiple trees this model based its decision on multiple images.


## Evaluation

The model produced approximately the following results:

* **Accuracy:** ~0.78–0.79
* **Sensitivity:** ~0.99
* **Specificity:** ~0.41–0.45
* **ROC-AUC:** ~0.94

The accuracy was relatively stable across runs. However, accuracy alone does not fully describe the model's performance because the dataset contains substantially more Pneumonia images than Normal images.

The model achieved **very high sensitivity (~0.99)**, meaning that it identified almost all of the Pneumonia cases in the test data.

However, its **specificity was considerably lower (~0.41–0.45)**. This means that the model had difficulty correctly identifying Normal cases and classified a substantial number of Normal images as Pneumonia.

The **ROC-AUC of approximately 0.94** indicates that the model was able to distinguish between the two classes well across different classification thresholds.

Therefore, the results show an important difference between the model's ability to separate the classes and its performance at the default 0.5 classification threshold.

## Conclusion

The Random Forest model demonstrated strong class-separation performance, as reflected by the ROC-AUC of approximately 0.94 and the high sensitivity of approximately 0.99.

However, the relatively low specificity indicates that the model frequently classified Normal images as Pneumonia when using the default classification threshold.

This suggests that the model's predictions should not be evaluated using accuracy alone. The classification threshold may also need to be examined depending on the intended application and the desired balance between detecting Pneumonia and correctly identifying Normal examinations.

The model therefore demonstrates potential as an experimental classification approach, but it would require further development and validation before it could be considered for use in an actual clinical workflow.

## Scope and Limitations

* The dataset is limited to pediatric patients between approximately one and five years old.
* The task is limited to binary classification between Pneumonia and Normal.
* The dataset contains substantially more Pneumonia images than Normal images, creating class imbalance.
* The dataset originates from a specific clinical population and institution, which may limit how well the model generalizes to other hospitals, populations, imaging equipment, and clinical settings.
* The validation set was not used because it contained only 16 images, which was considered insufficient for reliable validation of the model's performance. The current implementation used the training set for model fitting and the test set for final evaluation.
* A separate validation set was not necessary for the current basic Random Forest implementation, as no hyperparameter tuning or threshold optimization was performed using a validation set.

## Ethical Considerations

An automated classification model should be considered an **assistive tool rather than a replacement for clinical judgment**.

A positive prediction from the model should not be treated as a definitive diagnosis of Pneumonia. Final interpretation and diagnosis should remain under the responsibility of qualified healthcare professionals, particularly radiologists and physicians who can consider the patient's clinical history, physical examination, laboratory findings, and other relevant information.


---

## Technologies Used

* Python
* NumPy
* scikit-learn
* Matplotlib
* Pillow (PIL)
* Random Forest

---

## Author

**Johnnel V. Pangilinan**

Radiologic Technologist | Data Science Graduate Student

This project combines medical imaging experience with machine learning to explore applications of data science in healthcare.

---

## Disclaimer

This project is intended for **educational and research purposes only**.

It is not a medical device, diagnostic system, or substitute for professional medical evaluation. The model has not been clinically validated and should not be used to make patient-care decisions.


This is especially important for pediatric patients, where incorrect classification could potentially affect clinical decision-making. Therefore, any future clinical implementation would require appropriate validation, monitoring, human oversight, and consideration of patient safety and privacy.
