import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class Preprocessor:
    def __init__(
        self,
        target_column,
        all_target_columns,
        columns_to_drop,
        categorical_missing_token,
        numeric_imputation_strategy,
        categorical_imputation_strategy
    ):
        self.target_column = target_column
        self.all_target_columns = all_target_columns
        self.columns_to_drop = columns_to_drop
        self.categorical_missing_token = categorical_missing_token
        self.numeric_imputation_strategy = numeric_imputation_strategy
        self.categorical_imputation_strategy = categorical_imputation_strategy


    def clean_dataframe(self, df):

        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        final_rows = df.shape[0]

        if final_rows != initial_rows:
            print(f"Duplicated rows: {initial_rows - final_rows}")
            print(f"Dataframe shape: {df.shape}")

        target_to_drop = [
            col for col in self.all_target_columns
            if col != self.target_column
        ]
        print(f'{target_to_drop} target column eliminated')
        df = df.drop(columns=target_to_drop)

        df = df.drop(columns=self.columns_to_drop)
        print(f'{self.columns_to_drop} columns eliminated')


        # Não pode ser cat_col e se for object e dentro das colunas selecionadas fax X, caso contrário faz este código
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if self.categorical_missing_token is not None:
            df[cat_cols] = df[cat_cols].replace(
                self.categorical_missing_token,
                "Missing"
            )
        else:
            df[cat_cols] = df[cat_cols].fillna("Missing")    
            

        return df



    def fit(self, X):

        self.numeric_cols = X.select_dtypes(
            include=["int64", "float64"]
        ).columns
        print(f"{len(self.numeric_cols)} numeric features: {self.numeric_cols.tolist()}")

        self.categorical_cols = X.select_dtypes(
            include=["object", "category"]
        ).columns
        print(f"{len(self.categorical_cols)} categorical features: {self.categorical_cols.tolist()}")

        self.numeric_imputer = SimpleImputer(
            strategy=self.numeric_imputation_strategy
        )
        self.categorical_imputer = SimpleImputer(
            strategy=self.categorical_imputation_strategy
        )

        self.scaler = MinMaxScaler()
        self.ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        # Impute missing values
        X_num = self.numeric_imputer.fit_transform(X[self.numeric_cols])
        X_cat = self.categorical_imputer.fit_transform(X[self.categorical_cols])

        # Encoding 
        self.scaler.fit(X_num)
        self.ohe.fit(X_cat)

        return self



    def transform(self, X):

        # Impute missing values
        X_num = self.numeric_imputer.transform(X[self.numeric_cols])
        X_cat = self.categorical_imputer.transform(X[self.categorical_cols])

        # Encoding
        X_num_scaled = self.scaler.transform(X_num)
        X_cat_ohe = self.ohe.transform(X_cat)

        X_num_df = pd.DataFrame(
            X_num_scaled,
            columns=self.numeric_cols,
            index=X.index
        )

        X_cat_df = pd.DataFrame(
            X_cat_ohe,
            columns=self.ohe.get_feature_names_out(self.categorical_cols),
            index=X.index
        )

        X_final = pd.concat([X_num_df, X_cat_df], axis=1)

        X_final.columns = (
            X_final.columns
            .astype(str)
            .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
        )

        X_final = X_final.apply(pd.to_numeric, errors="raise")

        if not hasattr(self, "printed"):
            print(f"Total features after preprocessing: {X_final.shape[1]}")
            self.printed = True

        return X_final


