# def main():
#     print("Hello from auto-ml!")


# if __name__ == "__main__":
#     main()

from data_ingestion import DataIngestion

def main():
    file_path = "C:/Users/LENOVO/Desktop/Internship(Data_Science)/Project/Iris.csv" 
    target_variable = None

    ingestion = DataIngestion(file_path=file_path, target_variable=target_variable)

    if ingestion.df is not None:
        ingestion.generate_summary()
    else:
        print("Failed to load dataset.")

if __name__ == "__main__":
    main()