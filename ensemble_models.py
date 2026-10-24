"""Эксперимент с ансамблем или полносвязной нейронной сетью."""
from __future__ import annotations

from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


# German Credit stores these categories as integers, but the codes are not quantities.
CATEGORICAL_COLUMNS = {
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose",
    "Value_Savings_Stocks", "Length_of_current_employment", "Sex_Marital_Status",
    "Guarantors", "Most_valuable_available_asset", "Concurrent_Credits",
    "Type_of_apartment", "Occupation", "Telephone", "Foreign_Worker",
}


def categorical_columns(frame):
    return [col for col in frame.columns if col in CATEGORICAL_COLUMNS
            or not is_numeric_dtype(frame[col].dtype)]


def numeric_columns(frame):
    categories = categorical_columns(frame)
    return [col for col in frame.columns if col not in categories]


def make_preprocessor(features):
    if not hasattr(features, "columns"):
        return SimpleImputer(strategy="median")
    return ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric_columns),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_columns),
    ])


def train_and_evaluate(data: EnsembleInput):
    """Select by training-only cross-validation, then evaluate frozen candidates."""
    seed = data.random_seed
    candidates = {
        "Random Forest (leaf=2)": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "Extra Trees (leaf=2)": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1),
        "Gradient Boosting (depth=1)": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=1, random_state=seed),
        "Gradient Boosting (depth=2)": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=2, random_state=seed),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    experiments = []
    pipelines = []
    for name, classifier in candidates.items():
        model = Pipeline([("preprocess", make_preprocessor(data.train_features)),
                          ("model", classifier)])
        scores = cross_val_score(model, data.train_features, data.train_target,
                                 cv=cv, scoring="roc_auc", n_jobs=1, error_score="raise")
        experiments.append({"model_name": name, "cv_roc_auc_mean": float(scores.mean()),
                            "cv_roc_auc_std": float(scores.std()),
                            "cv_roc_auc_folds": scores.tolist(),
                            "parameters": classifier.get_params()})
        pipelines.append(model)

    # Selection is complete before test features or test labels are inspected.
    chosen = max(range(len(experiments)), key=lambda i: experiments[i]["cv_roc_auc_mean"])
    selected_result = None
    for i, (model, record) in enumerate(zip(pipelines, experiments)):
        model.fit(data.train_features, data.train_target)
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(data.test_features)[:, positive_index]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = {
            "roc_auc": float(roc_auc_score(data.test_target, probabilities)),
            "accuracy": float(accuracy_score(data.test_target, predictions)),
            "precision": float(precision_score(data.test_target, predictions, zero_division=0)),
            "recall": float(recall_score(data.test_target, predictions, zero_division=0)),
            "f1": float(f1_score(data.test_target, predictions, zero_division=0)),
        }
        record["test_metrics"] = metrics
        record["selected"] = i == chosen
        if i == chosen:
            selected_result = EnsembleResult(model_name=record["model_name"],
                                             probabilities=probabilities,
                                             predictions=predictions, **metrics)
    return selected_result, experiments


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучите выбранную модель и верните вероятности, метки и метрики на test.

    Разрешены Random Forest, Extra Trees, AdaBoost, Gradient Boosting,
    HistGradientBoosting, Voting/Stacking или полносвязная MLP. Не используйте
    test_target при обучении модели или подборе её параметров.
    """
    result, _ = train_and_evaluate(data)
    return result
