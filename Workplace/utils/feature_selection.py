import pandas as pd
from sklearn.feature_selection import mutual_info_classif


class MutualInformation:
    def __init__(
        self,
        threshold=0.05,
        random_state=80
    ):
        
        self.threshold = threshold
        self.random_state = random_state

        self.mutual_information_scores = None
        self.selected_features = None

    def fit(self, X, y):

        mi = mutual_info_classif(
            X,
            y,
            random_state=self.random_state
        )

        self.mutual_information_scores = (
            pd.Series(mi, index=X.columns)
            .sort_values(ascending=False)
        )

        selected = self.mutual_information_scores[self.mutual_information_scores >= self.threshold]

        self.selected_features = selected.index.tolist()
        print(f"Mutual Information selected {len(self.selected_features)} features: {self.selected_features}")

        return self

    def transform(self, X):
        return X[self.selected_features]


