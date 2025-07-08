import pandas as pd
from sklearn.decomposition import PCA


class DimensionalityReducer:
    """Class to model a dimensionality reduction method for the given dataset.
    1. Write a function to convert the high dimensional data to 2 dimensional.
    """

    def __init__(self, data: pd.DataFrame, n_components: int = 2):
        self.n_components = n_components
        self.data = data
        self.model = None
        self.transformed_data = None

    def transform(self):
        self.model = PCA(n_components=self.n_components)
        self.transformed_data = self.model.fit_transform(self.data)
        return pd.DataFrame(
            self.transformed_data,
            index=self.data.index,
            columns=[f"Component_{i+1}" for i in range(self.n_components)]
        )
