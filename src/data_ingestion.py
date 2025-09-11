import pandas as pd
import logging
from typing import List, Union
from tabulate import tabulate
from io import StringIO

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
        self.initial_summary = {}
        self.load_data()

    def load_data(self):
        logging.info(f"Ingesting data from {self.file_path}")
        try:
            file_path = self.file_path.lower()
            if file_path.endswith(".csv"):
                logging.info("Loading CSV file")
                self.df = pd.read_csv(self.file_path)
            elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
                logging.info("Loading Excel file")
                self.df = pd.read_excel(self.file_path)
            else:
                logging.warning(f"Unsupported file type: {self.file_path}")
                return
            
            # Validate target variable
            if self.target_variable and not self._validate_target_variable():
                print("Target Variable is not present in the dataset")
            
        except Exception as e:
            logging.error(f"Failed to load file {self.file_path}: {e}")
            self.df = None

    def _validate_target_variable(self) -> bool:
        """Checks whether the target variable exists in the dataset or not."""
        logging.info("Checking whether the output variable exists or not")
        if isinstance(self.target_variable, str):
            targets = [self.target_variable]
        else:
            targets = self.target_variable

        missing = [col for col in targets if col not in self.df.columns]
        if missing:
            logging.warning(f"Target variable(s) {missing} not found in dataset.")
            return False
        
        return True
    
    def generate_summary(self):
        if self.df is None:
            logging.error("No data loaded. Cannot generate summary.")
            return None

        # Capture df.info() into a string
        buffer = StringIO()
        self.df.info(buf=buffer)
        info_str = buffer.getvalue()

        # Handle inputs excluding targets (only if target is given)
        if self.target_variable:
            if isinstance(self.target_variable, str):
                inputs = [col for col in self.df.columns if col != self.target_variable]
            elif isinstance(self.target_variable, list):
                inputs = [col for col in self.df.columns if col not in self.target_variable]
            else:
                inputs = list(self.df.columns)
        else:
            inputs = None  # no target → no inputs

        self.initial_summary = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "inputs": inputs,
            "output": self.target_variable,
            "records_preview": self.df.head(),
            "statistical_information": self.df.describe(include="all"),
            "more_information": info_str
        }

        # === Pretty Console Output ===
        print("\n=== Dataset Summary ===")
        print(f"Rows: {self.initial_summary['rows']}")
        print(f"Columns: {self.initial_summary['columns']}")
        print(f"Target Variable(s): {self.initial_summary['output']}")

        print("\n=== Column Names ===")
        print(", ".join(self.initial_summary['column_names']))

        # Show inputs only if target_variable is provided
        if self.target_variable:
            print("\n=== Inputs (Features) ===")
            print(", ".join(self.initial_summary['inputs']))

        print("\n=== Records Preview ===")
        print(tabulate(self.initial_summary['records_preview'], headers="keys", tablefmt="pretty", showindex=False))

        print("\n=== Statistical Information ===")
        print(tabulate(self.initial_summary['statistical_information'], headers="keys", tablefmt="pretty"))

        print("\n=== DataFrame Info ===")
        print(self.initial_summary['more_information'])
        
        
# example usage

# from data_ingestion import DataIngestion

# def main():
#     df = DataIngestion("data/Iris.csv")
#     print(df.generate_summary())


# if __name__ == "__main__":
#     main()
