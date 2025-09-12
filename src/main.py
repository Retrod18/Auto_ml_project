from data_ingestion import DataIngestion

def main():
    df = DataIngestion("data/Iris.csv")
    df.generate_summary()
    


if __name__ == "__main__":
    main()
