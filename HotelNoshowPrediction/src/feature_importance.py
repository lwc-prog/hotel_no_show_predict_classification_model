import pandas as pd
import shap


class HotelNoShowFeatureImportance:
    """Extract feature importance from a fitted model pipeline."""

    def get_feature_importance(self, fitted_pipeline):
        model = fitted_pipeline.named_steps["model"]
        feature_names = fitted_pipeline.named_steps["preprocessor"].get_feature_names_out()

        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"{model.__class__.__name__} does not provide feature_importances_."
            )

        importance_df = pd.DataFrame(
            {
                "feature": feature_names,
                "importance": model.feature_importances_,
            }
        )

        return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)

    def get_top_features(self, fitted_pipeline, top_n=20):
        return self.get_feature_importance(fitted_pipeline).head(top_n)

    def get_shap_feature_importance(
        self,
        fitted_pipeline,
        X,
        sample_size=1000,
        random_state=42,
    ):
        """Rank features by mean absolute SHAP value."""
        shap_explanation = self.get_shap_explanation(
            fitted_pipeline,
            X,
            sample_size=sample_size,
            random_state=random_state,
        )

        importance_df = pd.DataFrame(
            {
                "feature": shap_explanation.feature_names,
                "mean_abs_shap": abs(shap_explanation.values).mean(axis=0),
            }
        )

        return importance_df.sort_values(
            "mean_abs_shap",
            ascending=False,
        ).reset_index(drop=True)

    def get_top_shap_features(
        self,
        fitted_pipeline,
        X,
        top_n=20,
        sample_size=1000,
        random_state=42,
    ):
        """Return the highest-ranked features using SHAP values."""
        return self.get_shap_feature_importance(
            fitted_pipeline,
            X,
            sample_size=sample_size,
            random_state=random_state,
        ).head(top_n)

    def get_shap_explanation(
        self,
        fitted_pipeline,
        X,
        sample_size=1000,
        random_state=42,
    ):
        """Create a SHAP explanation for the transformed model features."""
        model = fitted_pipeline.named_steps["model"]
        preprocessor = fitted_pipeline.named_steps["preprocessor"]
        X_sample = self._sample_rows(X, sample_size, random_state)
        X_transformed = preprocessor.transform(X_sample)
        X_transformed = self._as_dense_array(X_transformed)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)
        shap_values = self._positive_class_shap_values(shap_values)

        return shap.Explanation(
            values=shap_values,
            data=X_transformed,
            feature_names=feature_names,
        )

    def plot_shap_beeswarm(
        self,
        fitted_pipeline,
        X,
        sample_size=1000,
        random_state=42,
        max_display=20,
        show=True,
    ):
        """Display a SHAP beeswarm plot for the fitted model pipeline."""
        shap_explanation = self.get_shap_explanation(
            fitted_pipeline,
            X,
            sample_size=sample_size,
            random_state=random_state,
        )
        shap.plots.beeswarm(
            shap_explanation,
            max_display=max_display,
            show=show,
        )

        return shap_explanation

    def _sample_rows(self, X, sample_size, random_state):
        if sample_size is None or len(X) <= sample_size:
            return X

        return X.sample(n=sample_size, random_state=random_state)

    def _as_dense_array(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()

        return X

    def _positive_class_shap_values(self, shap_values):
        if isinstance(shap_values, list):
            return shap_values[1] if len(shap_values) > 1 else shap_values[0]

        if getattr(shap_values, "ndim", None) == 3:
            return shap_values[:, :, 1]

        return shap_values
