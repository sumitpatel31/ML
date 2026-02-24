from sklearn.model_selection import train_test_split
import pandas as pd
from encoding import encode_columns
def split_df():
    df=encode_columns()

    df = df.drop(columns=["Loan_ID"], errors="ignore")

    # encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    x=df.drop("Loan_Status",axis=1)
    y= df["Loan_Status"]
    #from sklearn.model_selection import train_test_split
    
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=1234
    )
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    return x_train, x_test, y_train, y_test

#split_df()