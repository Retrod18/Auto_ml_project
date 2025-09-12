import pandas as pd
from typing import List, Union

class DataIngestion:
    def __init__(self, file_path: str, target_variable: Union[List[str], str, None] = None):
        """
        Arguments:
        file_path - Path to the file (CSV/Excel).
        target_variable - It can be single or multiple target columns. By default it is None.
        kwargs - Additional arguments for pandas read functions.
        """
        self.file_path = file_path
        self.df = None
        self.target_variable = target_variable
        self.load_data()

    def load_data(self):
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                self.df = pd.read_excel(self.file_path)
            else:
                self.df = pd.read_csv(self.file_path)
                
            # validating target variable
            if self.target_variable:
                self.validate_target()
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")
            
    def validate_target(self):
        if isinstance(self.target_variable, str):
            self.target_variable = [self.target_variable]
        missing_cols = [col for col in self.target_variable if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Target variable(s) {missing_cols} not found in the dataframe columns.")
    
    def generate_summary(self):
        """
        Provides basic overview of the dataframe
        """
        if self.df is None:
            print("Cannot generate summary because no data is loaded.")
            return
            
        print("\n=== Dataset Summary ===")
        if self.target_variable:
            print(f"Target Variable(s): {self.target_variable}")
        print(f"Rows: {self.df.shape[0]}, Columns: {self.df.shape[1]}")
        print("\n=== Data Preview ===")
        print(self.df.head())
        print("\n=== Data Types & Non-Null Values ===")
        self.df.info()
        print("\n=== Statistical Summary ===")
        print(self.df.describe(include='all'))
        
# example usage

# from data_ingestion import DataIngestion

# def main():
#     df = DataIngestion("data/Iris.csv")
#     print(df.generate_summary())


# if __name__ == "__main__":
#     main()
