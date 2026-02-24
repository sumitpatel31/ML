import pandas as pd
import numpy as np  
import read_data


def cline( ):
    df=read_data.read_file()
    if df is not None:
        print("DataFrame loaded successfully:")
        
        # Converting 'Dependents' column values
        df["Dependents"]=df["Dependents"].replace("3+",3)
        df["Dependents"]=df["Dependents"].astype(float)

        # Separating categorical and numerical columns
        cat=df.select_dtypes(include="O").keys()
        num=df.select_dtypes(exclude="O").keys()

        for i in num:
            c = df[i]
            q1 = np.percentile(c, 25)
            q3 = np.percentile(c, 75)
            iqr = q3 - q1
            lb = q1 - (1.5 * iqr)
            ub = q3 + (1.5 * iqr)
            outlayer=(c<lb) | (c>ub)
            m=np.median(c)
            df.loc[outlayer,i]=m

        print("Data cleaning completed.")
        #print(df)
        return df

    else:
        print("Failed to load DataFrame.")


# data=cline()
# print(data.head())