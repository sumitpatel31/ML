try:
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier

    import train_test_Split

    df = pd.read_csv("loan_data.csv")  

    df = df.drop(columns=["Loan_ID"], errors="ignore")
    
    x_train, x_test, y_train, y_test=train_test_Split.split_df()
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)


    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }


    results = {}

    for name, model in models.items():

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

        results[name] = (acc, roc)

        print(f"\n{name}")
        print("Accuracy:", acc)
        print("ROC-AUC:", roc)
        print(classification_report(y_test, y_pred))

   
    best_model_name = max(results, key=lambda x: results[x][1])
    print("\nBest Model:", best_model_name)
    return best_model_name

except Exception as e:
    print("Error:", str(e))
