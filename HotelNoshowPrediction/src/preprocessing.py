from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import yaml


class HotelNoShowPreprocessor:
    """Preprocess hotel no-show data using the cleaning steps from EDA."""

    MONTH_TO_NUM = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    WORD_TO_NUM = {
        "one": 1,
        "two": 2,
    }
    DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

    def __init__(self, config_path=None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self.config = self._load_config()
        self.preprocessing_config = self.config.get("preprocessing", {})

        self.db_path = self._resolve_path(
            self.preprocessing_config.get("db_path", "../data/noshow.db")
        )
        self.table_name = self.preprocessing_config.get("table_name", "noshow")
        self.usd_to_sgd_rate = self.preprocessing_config.get("usd_to_sgd_rate", 1.35)
        self.date_year = self.preprocessing_config.get("date_year", 2024)

    def load_data(self):
        """Load raw data from the SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)

    def preprocess(self, df=None):
        """Run all preprocessing steps and return a clean dataframe."""
        df = self.load_data() if df is None else df.copy()

        df = self._drop_empty_rows(df)
        df = self._fix_checkout_dates(df)
        df = self._standardize_price(df)
        df = self._standardize_guest_counts(df)
        df = self._clean_categorical_columns(df)
        df = self._drop_engineered_source_columns(df)
        df = self._drop_low_value_columns(df)
        df = self._impute_price(df)

        return df.reset_index(drop=True)

    def _drop_empty_rows(self, df):
        """Drop rows where all fields except booking_id are missing."""
        non_id_columns = [col for col in df.columns if col != "booking_id"]
        return df.dropna(subset=non_id_columns, how="all").copy()

    def _fix_checkout_dates(self, df):
        """Repair negative checkout days and create days_difference."""
        checkout_day_abs = df["checkout_day"].abs()
        negative_checkout_mask = df["checkout_day"] < 0

        same_arrival_checkout_month = (
            self._clean_text(df["arrival_month"])
            == self._clean_text(df["checkout_month"])
        )
        arrival_before_checkout_abs = df["arrival_day"] < checkout_day_abs
        
        """Only consider checkout_day as invalid if it's negative, not in the same month as arrival, and not before the arrival day (after taking absolute value). This allows us to correct obvious data entry errors while retaining valid cases where checkout_day is negative but still logically consistent with the arrival date. See eda.ipynb for more details. """
        invalid_checkout_mask = (
            negative_checkout_mask
            & ~same_arrival_checkout_month
            & ~arrival_before_checkout_abs
        )

        df = df.loc[~invalid_checkout_mask].copy()
        df["checkout_day"] = df["checkout_day"].abs()
        df["days_difference"] = self._calculate_days_difference(df)

        return df

    def _calculate_days_difference(self, df):
        arrival_month = self._clean_text(df["arrival_month"]).map(self.MONTH_TO_NUM)
        checkout_month = self._clean_text(df["checkout_month"]).map(self.MONTH_TO_NUM)
        arrival_day = pd.to_numeric(df["arrival_day"], errors="coerce").astype("Int64")
        checkout_day = pd.to_numeric(df["checkout_day"], errors="coerce").astype("Int64")

        arrival_date = self._build_dates(arrival_month, arrival_day)
        checkout_date = self._build_dates(checkout_month, checkout_day)

        checkout_date = checkout_date.mask(
            checkout_date < arrival_date,
            checkout_date + pd.DateOffset(years=1),
        )

        return (checkout_date - arrival_date).dt.days

    def _build_dates(self, month, day):
        dates = pd.Series(pd.NaT, index=month.index, dtype="datetime64[ns]")
        valid_dates = month.notna() & day.notna()

        dates.loc[valid_dates] = pd.to_datetime(
            pd.DataFrame(
                {
                    "year": self.date_year,
                    "month": month.loc[valid_dates].astype(int),
                    "day": day.loc[valid_dates].astype(int),
                }
            ),
            errors="coerce",
        )

        return dates

    def _standardize_price(self, df):
        """Extract currency and amount, then convert prices to SGD."""
        price_text = df["price"].astype("string").str.strip()
        currency = price_text.str.extract(r"([A-Za-z]{3})", expand=False).str.lower()
        amount = pd.to_numeric(
            price_text.str.extract(r"(\d+(?:\.\d+)?)", expand=False),
            errors="coerce",
        )

        is_usd = currency.eq("usd").to_numpy(dtype=bool, na_value=False)
        is_sgd = currency.eq("sgd").to_numpy(dtype=bool, na_value=False)

        df["currency"] = currency
        df["price"] = amount
        df["price_sgd"] = np.select(
            [is_usd, is_sgd],
            [amount * self.usd_to_sgd_rate, amount],
            default=np.nan,
        )
        df["price_sgd"] = pd.Series(df["price_sgd"], index=df.index).round(2)

        return df

    def _standardize_guest_counts(self, df):
        """Convert guest count columns to nullable integer values."""
        num_adults = self._clean_text(df["num_adults"]).replace(self.WORD_TO_NUM)

        df["num_adults"] = pd.to_numeric(num_adults, errors="coerce").astype("Int64")
        df["num_children"] = pd.to_numeric(df["num_children"], errors="coerce").astype("Int64")

        return df

    def _clean_categorical_columns(self, df):
        categorical_columns = [
            "branch",
            "booking_month",
            "arrival_month",
            "checkout_month",
            "country",
            "first_time",
            "room",
            "platform",
        ]

        for col in categorical_columns:
            df[col] = self._clean_text(df[col])

        df["room"] = df["room"].fillna("missing")

        return df

    def _drop_engineered_source_columns(self, df):
        columns_to_drop = [
            "price",
            "currency",
            "arrival_day",
            "checkout_day",
        ]
        return df.drop(columns=columns_to_drop)

    def _drop_low_value_columns(self, df):
        columns_to_drop = [
            "checkout_month",
            "platform",
            "num_adults",
            "days_difference",
        ]
        return df.drop(columns=columns_to_drop)

    def _impute_price(self, df):
        """Impute missing price_sgd values using room median, then overall median."""
        df["price_sgd_missing"] = df["price_sgd"].isna().astype(int)

        room_median_price = df.groupby("room")["price_sgd"].transform("median")
        df["price_sgd_imputed"] = df["price_sgd"].fillna(room_median_price)
        df["price_sgd_imputed"] = df["price_sgd_imputed"].fillna(df["price_sgd"].median())

        return df.drop(columns=["price_sgd", "price_sgd_missing"])

    def _clean_text(self, series):
        return series.astype("string").str.strip().str.lower()

    def _load_config(self):
        with self.config_path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _resolve_path(self, path):
        path = Path(path)

        if path.is_absolute():
            return path

        return (self.config_path.parent / path).resolve()


if __name__ == "__main__":
    preprocessor = HotelNoShowPreprocessor()
    cleaned_df = preprocessor.preprocess()
    print("Final shape:", cleaned_df.shape)
    print("Columns:", cleaned_df.columns.tolist())
    print(cleaned_df.head())
    print(cleaned_df.dtypes)
