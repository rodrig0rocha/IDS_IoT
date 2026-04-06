import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualization:
    def __init__(self, df, label_columns):

        self.df = df
        self.label_columns = label_columns

        # Detect column types
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns

        # Exclude label columns
        self.numeric_cols = [c for c in numeric_cols if c not in label_columns]
        self.categorical_cols = [c for c in categorical_cols if c not in label_columns]



    def target_distribution(self, label_column, label_mapping=None):

        counts = self.df[label_column].value_counts().sort_values(ascending=False)
        percentages = counts / counts.sum() * 100

        # Apply mapping if provided
        if label_mapping is not None:
            x_labels = [label_mapping.get(v, str(v)) for v in counts.index]
        else:
            x_labels = counts.index.astype(str)

        plt.figure(figsize=(8, 5))
        plt.bar(x_labels, counts.values, color='#1f4e79')

        plt.title(f"Distribution of {label_column}")

        # Rotate labels to avoid overlap
        plt.xticks(rotation=35, ha="right")

        # Add vertical space for text
        max_count = counts.max()
        plt.ylim(0, max_count * 1.2)

        for i, (count, pct) in enumerate(zip(counts.values, percentages.values)):
            plt.text(
                i,
                count + max_count * 0.03,
                f"{pct:.2f}%",
                ha="center",
                va="bottom"
            )

        plt.tight_layout()
        plt.show()



    def numeric_distribution(self, column=None, bins=50):
        
        cols = [column] if column else self.numeric_cols

        for col in cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(self.df[col], bins=bins, kde=True, color='#1f4e79')
            plt.title(f"Distribution of {col} feature")
            plt.tight_layout()
            plt.show()



    def categorical_distribution(self, column=None, top_n=10):
        
        cols = [column] if column else self.categorical_cols

        for col in cols:
            counts = self.df[col].value_counts().head(top_n)

            plt.figure(figsize=(6, 4))
            counts.plot(kind="bar", color='#1f4e79')
            plt.title(f"Top {top_n} categories of {col} feature")
            plt.tight_layout()
            plt.show()



    def boxplot(self, column=None):
        
        cols = [column] if column else self.numeric_cols

        for col in cols:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=self.df[col], color="#1f4e79")
            plt.title(f"Boxplot of {col} feature")
            plt.tight_layout()
            plt.show()



    def scatter(self, x, y, hue=None):
        
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=self.df, x=x, y=y, hue=hue)
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.show()







