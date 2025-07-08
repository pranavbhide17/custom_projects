import pandas as pd
from sklearn.mixture import GaussianMixture


class DensityEstimator:
    def __init__(self, data: pd.DataFrame, dim_reducer, high_dim_feature_names):
        """
        :param data: 2D dataframe (after PCA)
        :param dim_reducer: fitted DimensionalityReducer instance
        :param high_dim_feature_names: original feature column names
        """
        self.data = data
        self.dim_reducer_model = dim_reducer.model
        self.feature_names = high_dim_feature_names
        self.gmm = None

    def fit_distribution(self, n_components=4):
        """
        Fit a Gaussian Mixture Model to the 2D PCA-reduced data
        :param n_components: number of GMM components/clusters
        """
        self.gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        self.gmm.fit(self.data)

    def sample_parties(self, n_samples=10) -> pd.DataFrame:
        """
        Sample synthetic parties in 2D space from the GMM distribution
        :param n_samples: how many to sample
        :return: DataFrame of shape (n_samples, 2)
        """
        if not self.gmm:
            raise ValueError("Model not fitted. Call fit_distribution() first.")
        samples, _ = self.gmm.sample(n_samples)
        return pd.DataFrame(samples, columns=self.data.columns)

    def inverse_transform_to_high_dim(self, sampled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Project 2D samples back to the high-dimensional policy space using PCA inverse
        :param sampled_df: DataFrame of shape (n_samples, 2)
        :return: DataFrame of shape (n_samples, original feature size)
        """
        return pd.DataFrame(
            self.dim_reducer_model.inverse_transform(sampled_df),
            columns=self.feature_names
        )
