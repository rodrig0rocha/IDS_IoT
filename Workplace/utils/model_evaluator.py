
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight



class ModelEvaluator:
    def __init__(
        self,
        test_size=0.2,
        random_state=80,
        task_type="binary",
        split_strategy="stratified",
        time_column=None
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.task_type = task_type
        self.split_strategy = split_strategy
        self.time_column = time_column

        self.results_df = None
        self.label_encoder = None
        self.class_map = None
        self.trained_models = {}


    def metrics(self, y_true, y_pred):
        avg = "binary" if self.task_type == "binary" else "macro"
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg, zero_division=0)
        }
    

    def split_dataset(self, X, y):

        if self.split_strategy == "temporal":

            temp_df = X.copy()
            temp_df["target"] = y

            temp_df = temp_df.sort_values(
                self.time_column
            ).reset_index(drop=True)

            split_idx = int(len(temp_df) * (1 - self.test_size))

            train_df = temp_df.iloc[:split_idx].copy()
            test_df = temp_df.iloc[split_idx:].copy()

            y_train = train_df["target"].values
            y_test = test_df["target"].values

            X_train = train_df.drop(
                columns=["target", self.time_column],
                errors="ignore"
            )

            X_test = test_df.drop(
                columns=["target", self.time_column],
                errors="ignore"
            )

            print(f"Train time range: {train_df[self.time_column].min()} - {train_df[self.time_column].max()}")
            print(f"Test time range: {test_df[self.time_column].min()} - {test_df[self.time_column].max()}")

            return X_train, X_test, y_train, y_test


        elif self.split_strategy == "stratified":

            return train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y 
            )



    def run(self, models, preprocessor, X, y, feature_selector=None):
        
        # Label Encoder
        if y.dtype == "object":
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.class_map = {
                cls: int(code)
                for cls, code in zip(
                    self.label_encoder.classes_,
                    self.label_encoder.transform(self.label_encoder.classes_)
                )
            }
            print(self.class_map)


        X_train, X_test, y_train, y_test = self.split_dataset(X, y)

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")


        # Preprocessing (Missing imputation and Encoding)
        preprocessor.fit(X_train)

        X_train_proc = preprocessor.transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # Feature selection
        if feature_selector is not None:

                feature_selector.fit(X_train_proc, y_train)
                X_train_proc = feature_selector.transform(X_train_proc)
                X_test_proc = feature_selector.transform(X_test_proc)


        results = []

        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train
        )

        for model_name, model in models.items():
            print(f"\nRunning model: {model_name}")               

            clf = clone(model)

            train_start = time.perf_counter()

            clf.fit(
                X_train_proc,
                y_train,
                sample_weight=sample_weight
            )

            training_time = time.perf_counter() - train_start

            y_train_pred = clf.predict(X_train_proc)

            inference_start = time.perf_counter()

            y_test_pred = clf.predict(X_test_proc)

            inference_time = time.perf_counter() - inference_start
            inference_time_per_sample_ms = (
            inference_time / len(X_test_proc)
            ) * 1000

            self.trained_models[model_name] = {
                "model": clf,
                "X_train": X_train_proc,
                "y_train": y_train,
                "X_test": X_test_proc,
                "y_test": y_test
            }

            train_metrics = self.metrics(
                y_train,
                y_train_pred
            )
            test_metrics = self.metrics(
                y_test, y_test_pred
            )



            # Confusion matrix
            cm = confusion_matrix(y_test, y_test_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            disp.plot(cmap='Blues', colorbar=False)
            plt.title(f"Confusion Matrix: {model_name}")
            plt.show()


            results.append({
                "model": model_name,
                "train_accuracy": train_metrics["accuracy"],
                "test_accuracy": test_metrics["accuracy"],
                "train_precision": train_metrics["precision"],
                "test_precision": test_metrics["precision"],
                "train_recall": train_metrics["recall"],
                "test_recall": test_metrics["recall"],
                "train_f1": train_metrics["f1"],
                "test_f1": test_metrics["f1"],
                "training_time_s": training_time,
                "inference_time_per_sample_ms": inference_time_per_sample_ms
            })

        self.results_df = (
            pd.DataFrame(results)
            .reset_index(drop=True)
        )

        return self.results_df


    def get_model_data(self, model_name):
        trained = self.trained_models[model_name]

        return (
            trained["model"],
            trained["X_train"],
            trained["y_train"],
            trained["X_test"],
            trained["y_test"]
        )
