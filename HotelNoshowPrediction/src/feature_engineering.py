from pathlib import Path

import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from preprocessing import HotelNoShowPreprocessor

class HotelNoShowFeatureEngineer:
    """Prepare preprocessed hotel no-show data for model training."""

    TARGET_COLUMN = "no_show"
    ID_COLUMNS = ["booking_id"]
    ORDINAL_COLUMNS = ["num_children"]
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.split_config = self.config.get("data_split", {})
        self.encoder_config = self.config.get("encoder", {})

        self.test_size = self.split_config.get("test_size", 0.2)
        self.random_state = self.split_config.get("random_state", 42)
        self.feature_columns = []
        self.categorical_columns = []
        self.ordinal_columns = []
        self.numeric_columns = []

    def prepare_data(self, df):
        """Split the dataframe into model features and target."""
        df = df.copy()
        df = df.dropna(subset=[self.TARGET_COLUMN])

        y = df[self.TARGET_COLUMN].astype(int)
        X = df.drop(columns=[self.TARGET_COLUMN] + self.ID_COLUMNS, errors="ignore")

        self.feature_columns = X.columns.tolist()
        self.categorical_columns = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        self.ordinal_columns = [col for col in self.ORDINAL_COLUMNS if col in X.columns]
        self.numeric_columns = [
            col
            for col in X.select_dtypes(include=["number", "bool"]).columns
            if col not in self.ordinal_columns
        ]

        return X, y

    def train_test_split(self, X, y):
        """Create a stratified train-test split."""
        stratify = y if self.split_config.get("stratify", True) else None

        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify,
        )

    def build_preprocessor(self):
        """Create the feature transformer used before model training."""
        return ColumnTransformer(
            transformers=[
                ("categorical", self._one_hot_encoder(), self.categorical_columns),
                ("ordinal", "passthrough", self.ordinal_columns),
                ("numeric", "passthrough", self.numeric_columns),
            ],
            remainder="drop",
        )

    def prepare_train_test_data(self, df):
        """Return X_train, X_test, y_train, and y_test."""
        X, y = self.prepare_data(df)
        return self.train_test_split(X, y)

    def get_feature_names(self, fitted_pipeline):
        """Return transformed feature names after fitting a pipeline."""
        preprocessor = fitted_pipeline.named_steps["preprocessor"]

        return preprocessor.get_feature_names_out()

    def _one_hot_encoder(self):
        handle_unknown = self.encoder_config.get("handle_unknown", "ignore")

        try:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown=handle_unknown, sparse=True)

    def _load_config(self, config_path):
        path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH

        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}


if __name__ == "__main__":
    preprocessor = HotelNoShowPreprocessor()
    df = preprocessor.preprocess()

    feature_engineer = HotelNoShowFeatureEngineer()
    X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_data(df)
    feature_preprocessor = feature_engineer.build_preprocessor()

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_train preview:")
    print(X_train.head())
    print("Target distribution:")
    print(y_train.value_counts(normalize=True).round(3))
    print("Categorical columns:", feature_engineer.categorical_columns)
    print("Ordinal columns:", feature_engineer.ordinal_columns)
    print("Numeric columns:", feature_engineer.numeric_columns)
    print("Feature preprocessor:", feature_preprocessor)
    
    X_train_encoded = feature_preprocessor.fit_transform(X_train)
    feature_names = feature_preprocessor.get_feature_names_out()
    X_train_encoded_preview = X_train_encoded[:5]

    if hasattr(X_train_encoded_preview, "toarray"):
        X_train_encoded_preview = X_train_encoded_preview.toarray()

    X_train_encoded_df = pd.DataFrame(X_train_encoded_preview, columns=feature_names)

    print("X_train encoded shape:", X_train_encoded.shape)
    print("X_train encoded preview:")
    print(X_train_encoded_df.head())
