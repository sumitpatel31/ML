
import train_test_Split
from model_development import dt, nb, knn, lr, rf



def m(model):
    x_train, x_test, y_train, y_test=train_test_Split.split_df()
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("y_pred :",y_pred.tolist())
    return y_test,y_pred

def predict():
    try:
        l = ["dt", "nb", "knn", "lr", "rf"]
        print("Choose one model from the list", l)
        name = input("enter hear: ")

        if name not in l:
            print("Enter valid input")
            return None, None

        if name == "dt":
            model = dt()
        elif name == "nb":
            model = nb()
        elif name == "knn":
            model = knn()
        elif name == "lr":
            model = lr()
        elif name == "rf":
            model = rf()

        print(f"You have selected: {name} model")
        return m(model)

    except Exception as e:
        print("Predict error:", e)
        return None, None

