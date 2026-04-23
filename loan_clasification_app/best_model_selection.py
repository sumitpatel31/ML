try:
    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    import train_test_Split


    class best_model:

        def __init__(self, file_path):
            self.file_path = file_path
            self.scaler = StandardScaler()
            self.models = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Naive Bayes": GaussianNB(),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            }
            self.results = {}

        def load_data(self):
            df = pd.read_csv(self.file_path)
            df = df.drop(columns=["Loan_ID"], errors="ignore")
            return df

        def split_data(self):
            x_train, x_test, y_train, y_test = train_test_Split.split_df()
            return x_train, x_test, y_train, y_test

        def scale_data(self, x_train, x_test):
            x_train_scaled = self.scaler.fit_transform(x_train)
            x_test_scaled = self.scaler.transform(x_test)
            return x_train_scaled, x_test_scaled

        def train_and_evaluate(self):

            x_train, x_test, y_train, y_test = self.split_data()
            x_train_scaled, x_test_scaled = self.scale_data(x_train, x_test)

            for name, model in self.models.items():

                if name in ["Logistic Regression", "KNN", "Naive Bayes"]:
                    model.fit(x_train_scaled, y_train)
                    y_pred = model.predict(x_test_scaled)
                    y_prob = model.predict_proba(x_test_scaled)[:, 1]
                else:
                    model.fit(x_train, y_train)
                    y_pred = model.predict(x_test)
                    y_prob = model.predict_proba(x_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_prob)

                self.results[name] = (acc, roc)

                print(f"\n{name}")
                print("Accuracy:", acc)
                print("ROC-AUC:", roc)
                print(classification_report(y_test, y_pred))

            best_model_name = max(self.results, key=lambda x: self.results[x][1])
            return best_model_name

except Exception as e:
    print("Error:", str(e))
