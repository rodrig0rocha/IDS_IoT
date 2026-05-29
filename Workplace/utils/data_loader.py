from pathlib import Path
import pandas as pd

pd.set_option('display.max_columns', None)

class DatasetLoader:

    dataset_root = Path("../Datasets")


    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.dataset_path = self.dataset_root / dataset_name

        if not self.dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset folder not found: {self.dataset_path}"
            )

        self.files, self.file_type = self.search_files()

        if not self.files:
            raise FileNotFoundError(
                f"No files found in {self.dataset_path}"
            )


    def search_files(self):

        csv_files = sorted(self.dataset_path.glob("*.csv"))
        xlsx_files = sorted(self.dataset_path.glob("*.xlsx"))

        if csv_files:
            return csv_files, "csv"
        
        if xlsx_files:
            return xlsx_files, "xlsx"
        
        return [], None


    def load(self):

        dfs = []

        for f in self.files:

            if self.file_type== "csv":
                df = pd.read_csv(f, low_memory=False)

            elif self.file_type == "xlsx":
                df = pd.read_excel(f)

            dfs.append(df)        

        return pd.concat(dfs, ignore_index=True)


    def info(self):
    
        df = self.load()

        temp = pd.DataFrame(index=df.columns)

        temp["Datatype"]   = df.dtypes
        temp["Not nulls"]  = df.count()
        temp["Nulls"]      = df.isnull().sum()
        temp["% Nulls"]    = df.isnull().mean() * 100
        temp["Unique cnt"] = df.nunique()

        numeric_cols = df.select_dtypes(include=["int64", "float64"])

        temp["Mean"]  = numeric_cols.mean()
        temp["StDev"] = numeric_cols.std()
        temp["Min"]   = numeric_cols.min()
        temp["Q25"]   = numeric_cols.quantile(0.25)
        temp["Q50"]   = numeric_cols.quantile(0.50)
        temp["Q75"]   = numeric_cols.quantile(0.75)
        temp["Max"]   = numeric_cols.max()

        return temp
    

    def remove_duplicate(self, df):

        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        final_rows = df.shape[0]

        print(f"Duplicated rows: {initial_rows - final_rows}")
        print(f"Dataframe shape: {df.shape}")


        return df  
