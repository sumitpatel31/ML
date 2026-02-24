import pandas as pd
import missing_impuation

from sklearn.preprocessing import OrdinalEncoder




def encode_columns():

    
    df=missing_impuation.impute_missing_values()

    one=["Gender","Married"]
    ordinal_cols=['Education', 'Self_Employed',
        'Property_Area']
    labal=["Loan_Status"]


    try:
        #ordinal_cols = ['Education', 'Self_Employed', 'Property_Area']

        encoder = OrdinalEncoder(
            categories=[
                ['Not Graduate', 'Graduate'],      # Education
                ['No', 'Yes'],                     # Self_Employed
                ['Rural', 'Semiurban', 'Urban']    # Property_Area
            ]
        )

        df[ordinal_cols] = encoder.fit_transform(df[ordinal_cols])

        arr=labal+one
        for i in arr:
            label=sorted(df[i].astype(str).unique())
            d={label[i]:i for i in range(len(label))}
            df[i]=df[i].map(d)

        print("Ordinal encoding completed.")
        
        return df


    except Exception as e:
        print(f"Error during ordinal encoding: {e}")


#encode_columns()
