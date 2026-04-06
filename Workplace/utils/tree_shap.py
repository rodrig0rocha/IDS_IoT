import shap

shap.initjs()

class SHAPTreeExplainer:
    def __init__(
        self, 
        task_type="binary", 
        sample_size=1000, 
        random_state=80
    ):
        self.task_type = task_type
        self.sample_size = sample_size
        self.random_state = random_state


    def explain(self, model, X):

        explainer = shap.TreeExplainer(model)

        X_sample = X.sample(self.sample_size, random_state=self.random_state)
        shap_values = explainer(X_sample)

        return shap_values, X_sample


    def plot_beeswarm(self, model, X, class_index=0):

        shap_values, X_sample = self.explain(model, X)

        if self.task_type == "binary":
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  

            shap.plots.beeswarm(shap_values, max_display=10)

        elif self.task_type == "multiclass":
            shap.plots.beeswarm(shap_values[:, :, class_index], max_display=10)


            



