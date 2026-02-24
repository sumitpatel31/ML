import pandas as pd
import data_clean



def impute_missing_values():
    # Handling missing values
    try:
        df=data_clean.cline()
        d=df.isnull().sum()
        nl=[]
        for i,c in d.items():
            if c!=0:
                nl.append(i)
        for i in nl:
            if df[i].dtype !="O":
                median=df[i].median()
                df[i]=df[i].fillna(median)
            else:
                mode=df[i].mode()[0]
                df[i]=df[i].fillna(mode)
        print("Missing value imputation completed.")
        #print(df.isnull().sum())
        # print(df.head())
        return df
    except Exception as e:
        print(f"Error during missing value imputation: {e}")
        return None

# impute_missing_values()