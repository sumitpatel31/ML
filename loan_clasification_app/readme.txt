# 🚀 Loan Prediction ML API (FastAPI)

A production-ready Machine Learning API built using **FastAPI** that performs loan approval prediction using multiple ML models.

This project includes:
- Complete ML preprocessing pipeline
- Model training & evaluation
- CSV file upload support
- Multiple model selection
- Random Forest with feature importance
- Clean modular architecture

---

## 📌 Tech Stack

- Python 3.12
- FastAPI
- Scikit-Learn
- Pandas
- NumPy
- Uvicorn

---

## ⚙️ Features

✅ Modular ML pipeline  
✅ Upload CSV dataset via API  
✅ Train & evaluate models dynamically  
✅ Supported models:
- Decision Tree
- Naive Bayes
- KNN
- Logistic Regression
- Random Forest  

✅ Returns:
- Best Model Name
- Accuracy
- Precision
- Recall
- F1 Score
- (Optional) Feature Importance (Random Forest)

---

## 🚀 How to Run the Project

    1. Install dependencies
    pip install fastapi uvicorn pandas numpy scikit-learn

    2. Start the FastAPI server
    uvicorn app:app --reload
