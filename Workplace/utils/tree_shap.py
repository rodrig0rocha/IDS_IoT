import shap

shap.initjs()

class SHAPTreeExplainer:
    def __init__(
        self, 
        task_type="binary",
        n_samples=None,
        random_state=None
    ):
        self.task_type = task_type
        self.n_samples = n_samples
        self.random_state = random_state


    def subsample(self, X):
        if self.n_samples is None:
            return X
        
        return X.sample(self.n_samples, random_state=self.random_state)


    def explain(self, model, X):

        X = self.subsample(X)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)

        return shap_values, X


    def plot_beeswarm(self, model, X, class_index=0):

        shap_values, X = self.explain(model, X)

        if self.task_type == "binary":
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  

            shap.plots.beeswarm(shap_values, max_display=10)

        elif self.task_type == "multiclass":
            shap.plots.beeswarm(shap_values[:, :, class_index], max_display=10)

