from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def ensemble_models(task_type, random_state=80):

    models = {

        # Bagging
        "RandomForest": RandomForestClassifier(
            random_state=random_state
        ),

        "ExtraTrees": ExtraTreesClassifier(
            random_state=random_state
        ),

        # Boosting
        "LightGBM": LGBMClassifier(
            random_state=random_state,
            verbosity=-1
        )
    }


    if task_type == "binary":

        models["XGBoost"] = XGBClassifier(
            random_state=random_state,
            objective="binary:logistic",
            eval_metric="logloss"
        )

    else:

        models["XGBoost"] = XGBClassifier(
            random_state=random_state,
            objective="multi:softprob",
            eval_metric="mlogloss"
        )


    return models
