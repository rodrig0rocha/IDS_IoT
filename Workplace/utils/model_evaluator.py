
import time
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


class ModelEvaluator:
    def __init__(
        self,
        test_size=0.2,
        random_state=80,
        task_type="binary"
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.task_type = task_type

        self.results_df = None
        self.label_encoder = None
        self.class_map = None
        self.trained_models = {}


    def metrics(self, y_true, y_pred):
        avg = "binary" if self.task_type == "binary" else "weighted"
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
            "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
            "f1": f1_score(y_true, y_pred, average=avg, zero_division=0)
        }


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

        # Split Dataset 
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task_type == "binary" else None
        )

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

        for model_name, model in models.items():
            print(f"\nRunning model: {model_name}")               

            X_train_final, y_train_final = SMOTE(
                random_state=self.random_state
            ).fit_resample(X_train_proc, y_train)

            start_time = time.time() 

            clf = clone(model)
            clf.fit(X_train_final, y_train_final)

            self.trained_models[model_name] = {
                "model": clf,
                "X_train": X_train_final,
                "y_train": y_train_final,
                "X_test": X_test_proc,
                "y_test": y_test
            }

            train_metrics = self.metrics(
                y_train_final, clf.predict(X_train_final)
            )
            test_metrics = self.metrics(
                y_test, clf.predict(X_test_proc)
            )

            elapsed = time.time() - start_time

            # Confusion matrix
            y_test_pred = clf.predict(X_test_proc)
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
                "time": elapsed
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
