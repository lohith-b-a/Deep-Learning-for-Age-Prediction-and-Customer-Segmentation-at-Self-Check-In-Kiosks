# Deep Learning for Age Prediction and Customer Segmentation at Self-Check-In Kiosks

## Project Overview

This project uses computer vision and deep learning to build an automated age verification system for self-check-in kiosks in retail stores. The model predicts a person's age and classifies them into age-based groups (e.g., senior citizens, students) to enable regulatory compliance and personalized discount offerings.

The model was trained using the [UTKFace dataset](https://susanqq.github.io/UTKFace/) and deployed in a scenario inspired by **Lidl**, a European supermarket chain seeking to implement real-time age detection.

---

## Business Problem

Retail chains are deploying self-check-in kiosks to enhance customer experience and reduce checkout friction. However, they must still comply with regulations regarding age-restricted products (e.g., alcohol, tobacco).  
Traditional manual ID checks don't fit well with self-checkout automation. A scalable solution is needed.

Additionally, the retailer wants to offer **age-based discounts and promotions** (e.g., senior, student, child-specific discounts). This requires an accurate and fast way to classify customers by age group.

---

## Dataset Summary

- **Source**: UTKFace Dataset (23,991 labeled images)
- **Format**: Images named in the format AGE_GENDER_ETHNICITY_TIMESTAMP.jpg
- **Label Extraction**: Age (0–116), Gender (0 = Male, 1 = Female)
- **Image Size**: Resized to 64×64 and normalized

---

## Problem Framing

- **Age Prediction**: Regression task (continuous target)
- **Age Group Classification**: Multi-class classification (5 age categories)
  - 0–12 → Infants & Children
  - 13–17 → Teenagers
  - 18–34 → Students & Young Professionals
  - 35–64 → Adults
  - 65+ → Seniors

---

## Workflow Summary

1. **Data Acquisition & Cleaning**
   - Downloaded and cleaned image files
   - Parsed filenames to extract age and gender
   - Converted images to pixel arrays using OpenCV

2. **Feature Engineering**
   - Created age group labels for classification
   - Normalized pixel values
   - Resized all images to 64×64×3

3. **Model Building**
   - Built custom Convolutional Neural Network (CNN)
   - Trained two models:
     - Regression model for age prediction
     - Classification model for age category

4. **Evaluation**
   - Used MAE, MSE for regression
   - Used accuracy, confusion matrix, and classification report for classification

---

## Tools & Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, pandas
- matplotlib, seaborn
- Jupyter Notebook

---

## Key Outcomes

- Built an end-to-end deep learning pipeline for facial image-based age prediction
- Achieved accurate age classification into business-relevant categories
- Enables use in retail environments for:
  - Automated age verification (regulatory compliance)
  - Age-based discount offers (targeted marketing)

---

## Author

**Lohith Basavanahalli Anjinappa**  

