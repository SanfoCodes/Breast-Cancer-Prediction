
# Breast Cancer Prediction: Classification Models and Neural Networks

This project compares two approaches for breast cancer prediction:
1. **Approach 1**: Traditional machine learning classification models (Logistic Regression, Random Forest, SVM, XGBoost).
2. **Approach 2**: Neural Networks using TensorFlow/Keras.

The goal is to evaluate the performance of both approaches on multiple breast cancer datasets and identify the best-performing model.

---

## **Datasets**
The following datasets are used in this project:
1. **Wisconsin Breast Cancer Dataset**:
   - Features: Numerical (e.g., radius, texture, perimeter).
   - Target: Binary diagnosis (malignant/benign).

2. **Breast Cancer Dataset**:
   - Features: Numerical (e.g., age, tumor size) and categorical (e.g., menopause, metastasis).
   - Target: Binary diagnosis.

3. **BRCA Dataset**:
   - Features: Numerical (e.g., age, protein levels) and categorical (e.g., tumor stage, surgery type).
   - Target: Binary patient status.

4. **German Breast Cancer Dataset**:
   - Features: Numerical (e.g., age, tumor size) and categorical (e.g., hormonal status).
   - Target: Binary status.

5. **SEER Breast Cancer Dataset**:
   - Features: Numerical (e.g., age, survival months) and categorical (e.g., tumor stage, estrogen status).
   - Target: Binary status.

---

## **Approach 1: Classification Models**
### **Models Used**
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- XGBoost

### **Steps**
1. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical features using `StandardScaler`.
   - Split data into training and testing sets.

2. **Model Training**:
   - Train each model on the training data.

3. **Model Evaluation**:
   - Evaluate models using **Accuracy**, **ROC-AUC**, and **Confusion Matrix**.
   - Combine predictions using a **weighted average** to create a fused model.

4. **Visualization**:
   - Plot ROC curves and bar plots for accuracy and AUC across datasets.

---

## **Approach 2: Neural Networks**
### **Model Used**
- A **Sequential Neural Network** with:
  - Input layer: 128 neurons, ReLU activation.
  - Hidden layers: 64 and 32 neurons, ReLU activation.
  - Output layer: Softmax (for multiclass) or Sigmoid (for binary classification).

### **Steps**
1. **Data Preprocessing**:
   - Handle missing values.
   - Encode categorical variables.
   - Scale numerical features using `StandardScaler`.
   - Split data into training and testing sets.

2. **Handle Class Imbalance**:
   - Use **SMOTE** (Synthetic Minority Oversampling Technique) to balance the dataset.
   - Compute class weights for training.

3. **Model Training**:
   - Train the neural network using the balanced dataset.
   - Use **Binary Crossentropy** (for binary classification) or **Categorical Crossentropy** (for multiclass classification) as the loss function.
   - Optimizer: Adam with a learning rate of 0.001.

4. **Model Evaluation**:
   - Evaluate the model using **Accuracy**, **ROC-AUC**, and **Confusion Matrix**.
   - For multiclass classification, use **Weighted ROC-AUC**.

5. **Visualization**:
   - Plot accuracy and ROC curves for each dataset.

---

## **Evaluation Metrics**
### **1. Accuracy**
- Measures the proportion of correctly predicted instances.
### **2. ROC-AUC**
- Measures the model's ability to distinguish between classes across all thresholds.
### **3. F1 Score**
- Harmonic mean of Precision and Recall.

---

## **Results**
- **Approach 1 (Classification Models)**:
  - Best-performing model: XGBoost (highest accuracy and AUC).
  - Fused model (weighted average) outperforms individual models.

- **Approach 2 (Neural Networks)**:
  - Achieves competitive performance, especially on imbalanced datasets.
  - SMOTE and class weighting improve model performance.

---

## **How to Run the Code**
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts:
   - For Approach 1 (Classification Models):
     ```bash
     python classification_models.py
     ```
   - For Approach 2 (Neural Networks):
     ```bash
     python neural_network.py
     ```

4. View results:
   - ROC curves and accuracy plots are saved in the `results/` folder.

---

## **Dependencies**
- Python 3.x
- Libraries:
  - `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `imblearn`, `matplotlib`, `seaborn`, `xgboost`

---

## **Conclusion**
- Both approaches perform well, but **Approach 1 (Classification Models)** is faster and easier to interpret.
- **Approach 2 (Neural Networks)** is more powerful for complex datasets but requires more computational resources.


## **Author**
- [Arpitha Thippeswamy]
- Contact: [arpitha.thipeswamy@gmail.com]

---

