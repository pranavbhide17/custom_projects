from pathlib import Path
from typing import List
from urllib.request import urlretrieve
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DataLoader:
    """Class to load the political parties dataset"""

    data_url: str = "https://www.chesdata.eu/s/CHES2019V3.dta"

    def __init__(self):
        self.party_data = self._download_data()
        self.non_features = []
        self.index = ["party_id", "party", "country"]

    def _download_data(self) -> pd.DataFrame:
        data_path, _ = urlretrieve(
            self.data_url,
            Path(__file__).parents[2].joinpath(*["data", "CHES2019V3.dta"]),
        )
        return pd.read_stata(data_path)

    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Write a function to remove duplicates in a dataframe"""
        return df.drop_duplicates(subset=["party_id"])

    def remove_nonfeature_cols(
        self, df: pd.DataFrame, non_features: List[str], index: List[str]
    ) -> pd.DataFrame:
        self.non_features = non_features
        self.index = index
        df = df.drop(columns=non_features, errors='ignore')
        df = df.set_index(index)
        return df

    def handle_NaN_values(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.fillna(df.median(numeric_only=True))

    def scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, index=df.index, columns=df.columns)

    def preprocess_data(self):
        df = self.party_data.copy()
        df = self.remove_duplicates(df)

        non_features = [
            "eu_googov_require", "eu_political_require", "eu_econ_require"
        ]
        index = ["party_id", "party", "country"]

        df = self.remove_nonfeature_cols(df, non_features, index)
        df = self.handle_NaN_values(df)
        df = self.scale_features(df)
        return df
