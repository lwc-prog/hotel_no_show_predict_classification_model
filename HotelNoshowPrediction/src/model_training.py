from pathlib import Path

import yaml
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from feature_engineering import HotelNoShowFeatureEngineer
from preprocessing import HotelNoShowPreprocessor


class HotelNoShowModelTrainer:
    """Build, train, and evaluate hotel no-show classification models."""

    SUPPORTED_MODELS = ("extra_trees", "random_forest", "xgboost")
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

    def __init__(self, feature_engineer, config_path=None):
        self.feature_engineer = feature_engineer
        self.config = self._load_config(config_path)
        self.model_config = self.config.get("models", {})
        self.metrics_config = self.config.get("metrics", {})
        self.random_state = self.config.get("data_split", {}).get("random_state", 42)

    def build_model(self, model_name):
        """Create one of the selected classification models."""
        self._validate_model_name(model_name)

        if model_name == "extra_trees":
            return ExtraTreesClassifier(**self._model_params(model_name))

        if model_name == "random_forest":
            return RandomForestClassifier(**self._model_params(model_name))

        return XGBClassifier(**self._model_params(model_name))

    def build_pipeline(self, model_name):
        """Build a full training pipeline for a selected model."""
        return Pipeline(
            steps=[
                ("preprocessor", self.feature_engineer.build_preprocessor()),
                ("model", self.build_model(model_name)),
            ]
        )

    def build_all_pipelines(self):
        """Build pipelines for Extra Trees, Random Forest, and XGBoost."""
        return {
            model_name: self.build_pipeline(model_name)
            for model_name in self.SUPPORTED_MODELS
        }

    def train_model(self, model_name, X_train, y_train):
        """Fit one configured model pipeline."""
        pipeline = self.build_pipeline(model_name)
        pipeline.fit(X_train, y_train)
        return pipeline

    def train_all_models(self, X_train, y_train):
        """Fit all configured model pipelines."""
        return {
            model_name: self.train_model(model_name, X_train, y_train)
            for model_name in self.SUPPORTED_MODELS
        }

    def evaluate_model(self, fitted_pipeline, X_test, y_test):
        """Evaluate a fitted pipeline using configured classification metrics."""
        y_pred = fitted_pipeline.predict(X_test)
        y_score = self._prediction_scores(fitted_pipeline, X_test)

        metric_functions = {
            "accuracy": lambda: accuracy_score(y_test, y_pred),
            "precision": lambda: precision_score(y_test, y_pred, zero_division=0),
            "recall": lambda: recall_score(y_test, y_pred, zero_division=0),
            "f1": lambda: f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": lambda: roc_auc_score(y_test, y_score) if y_score is not None else None,
        }

        return {
            metric: metric_functions[metric]()
            for metric in self.get_metrics()
            if metric in metric_functions
        }

    def get_metrics(self):
        """Return configured model evaluation metrics."""
        return self.metrics_config.get(
            "scoring",
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
        )

    def get_primary_metric(self):
        """Return the metric used for selecting the best model."""
        return self.metrics_config.get("primary", "f1")

    def _model_params(self, model_name):
        params = self.model_config.get(model_name, {}).copy()
        params.setdefault("random_state", self.random_state)
        return params

    def _validate_model_name(self, model_name):
        if model_name not in self.SUPPORTED_MODELS:
            valid_names = ", ".join(self.SUPPORTED_MODELS)
            raise ValueError(f"model_name must be one of: {valid_names}")

    def _prediction_scores(self, fitted_pipeline, X_test):
        if hasattr(fitted_pipeline, "predict_proba"):
            return fitted_pipeline.predict_proba(X_test)[:, 1]

        if hasattr(fitted_pipeline, "decision_function"):
            return fitted_pipeline.decision_function(X_test)

        return None

    def _load_config(self, config_path):
        path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}


if __name__ == "__main__":
    preprocessor = HotelNoShowPreprocessor()
    df = preprocessor.preprocess()

    feature_engineer = HotelNoShowFeatureEngineer()
    X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_data(df)

    trainer = HotelNoShowModelTrainer(feature_engineer)
    pipelines = trainer.build_all_pipelines()

    print("Available pipelines:", list(pipelines))
    print("Primary metric:", trainer.get_primary_metric())
    print("Scoring metrics:", trainer.get_metrics())
