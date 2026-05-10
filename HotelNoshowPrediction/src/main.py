import pandas as pd

from feature_engineering import HotelNoShowFeatureEngineer
from feature_importance import HotelNoShowFeatureImportance
from model_training import HotelNoShowModelTrainer
from preprocessing import HotelNoShowPreprocessor


def main():
    print("Loading and preprocessing data...")
    preprocessor = HotelNoShowPreprocessor()
    df = preprocessor.preprocess()
    print("Preprocessed data shape:", df.shape)

    print("\nPreparing train-test data...")
    feature_engineer = HotelNoShowFeatureEngineer()
    X_train, X_test, y_train, y_test = feature_engineer.prepare_train_test_data(df)
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Target distribution in training data:")
    print(y_train.value_counts(normalize=True).round(3))

    trainer = HotelNoShowModelTrainer(feature_engineer)
    primary_metric = trainer.get_primary_metric()

    print("\nTraining and evaluating models...")
    results = []
    fitted_pipelines = {}

    for model_name in trainer.SUPPORTED_MODELS:
        print(f"Training {model_name}...")
        fitted_pipeline = trainer.train_model(model_name, X_train, y_train)
        fitted_pipelines[model_name] = fitted_pipeline
        metrics = trainer.evaluate_model(fitted_pipeline, X_test, y_test)
        results.append({"model": model_name, **metrics})

    results_df = pd.DataFrame(results).sort_values(
        by=primary_metric,
        ascending=False,
    )

    best_model = results_df.iloc[0]

    print("\nModel performance:")
    print(results_df.round(4).to_string(index=False))

    print("\nConclusion:")
    print(
        f"The best model is {best_model['model']} based on "
        f"{primary_metric} = {best_model[primary_metric]:.4f}."
    )

    importance_analyzer = HotelNoShowFeatureImportance()
    best_model_name = best_model["model"]
    feature_importance_df = importance_analyzer.get_top_features(
        fitted_pipelines[best_model_name],
        top_n=10,
    )

    print(f"\nTop 10 feature importance for best model ({best_model_name}):")
    print(feature_importance_df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
