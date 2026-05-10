import pandas as pd


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
