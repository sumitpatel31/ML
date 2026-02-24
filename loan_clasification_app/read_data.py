import pandas as pd
file_path='loan_data.csv'
def read_file( ):
   
    try:
        data = pd.read_csv(file_path)
        df=pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return None 