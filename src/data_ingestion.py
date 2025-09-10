import pandas as pd
import logging
from typing import List

class DataIngestion:
    def __init__(self, file_path: str, target_variable: List[str] | str | None = None, **kwargs):
        """
        Arguments:
        file_path - Path to the file (CSV/Excel).
        target_variable - It can be single or multiple target columns By default it is None.
        """
        self.file_path = file_path
        self.df = None
        self.target_variable = target_variable
        self.initial_summary = {}
        self.read_kwargs = kwargs # additional arguments (optional)
        self.load_data()

    def load_data(self):
        logging.info(f"Ingesting data from {self.file_path}")
        try:
            file_path = self.file_path
            if file_path.endswith(".csv"):
                logging.info("Loading CSV file")
                self.df = pd.read_csv(self.file_path)
            elif file_path.endswith(".xlsx"):
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
        # checks whether the target variable exists in the dataset or not
        logging.info("chekcing whether the output variable exists or not")
        if isinstance(self.target_variable, str):
            logging.info("One output to be found")
            targets = [self.target_variable]
        else:
            logging.info("Multiple outputs to be found")
            targets = self.target_variable

        missing = [col for col in targets if col not in self.df.columns]
        if missing:
            logging.warning(f"Target variable(s) {missing} not found in dataset.")
            return False
        
        return True
    
    def generate_summary(self):
        # Build dataset summary
        self.initial_summary = {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "column_names": list(self.df.columns),
            "inputs": ", ".join([col for col in self.df.columns if col != self.target_variable]),
            "output": self.target_variable,
            "records_preview": self.df.head(),
            "statistical_information": self.df.describe(),
            "more_information": self.df.info()
        }