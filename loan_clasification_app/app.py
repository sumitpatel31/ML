from fastapi import FastAPI
from pydantic import BaseModel
import traceback

from train_test_Split import split_df
from model_devlopment import dt, nb, knn, lr, rf
from sklearn.metrics import accuracy_score

app = FastAPI(title="Loan Prediction ML API")

# -----------------------------
# Global objects (loaded once)
# -----------------------------
X_train, X_test, y_train, y_test = None, None, None, None
models = {}

# -----------------------------
# Startup event
# -----------------------------
@app.on_event("startup")
def load_data_and_models():
    global X_train, X_test, y_train, y_test, models
    try:
        X_train, X_test, y_train, y_test = split_df()

        models = {
            "decision_tree": dt(),
            "naive_bayes": nb(),
            "knn": knn(),
            "logistic_regression": lr(),
            "random_forest": rf()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)

        print("Models trained successfully.")

    except Exception as e:
        print("Startup error:", e)
        traceback.print_exc()


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {"message": "Loan Prediction API is running 🚀"}


@app.get("/models")
def list_models():
    return {"available_models": list(models.keys())}


@app.get("/evaluate/{model_name}")
def evaluate_model(model_name: str):
    try:
        if model_name not in models:
            return {"error": "Model not found"}

        model = models[model_name]
        preds = model.predict(X_test)

        acc = round(accuracy_score(y_test, preds), 3)

        return {
            "model": model_name,
            "accuracy": acc
        }

    except Exception as e:
        return {
            "error": str(e)
        }
