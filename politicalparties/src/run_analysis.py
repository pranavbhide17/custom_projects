from pathlib import Path
from matplotlib import pyplot

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import (
    scatter_plot,
    plot_density_estimation_results,
    plot_finnish_parties,
)

if __name__ == "__main__":

    # Step 1: Load and preprocess data
    data_loader = DataLoader()
    processed_data = data_loader.preprocess_data()

    # Step 2: Dimensionality reduction (PCA)
    dim_reducer = DimensionalityReducer(processed_data)
    reduced_dim_data = dim_reducer.transform()

    # Step 3: Visualize PCA-reduced data
    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(
        reduced_dim_data,
        color="r",
        splot=splot,
        label="PCA-reduced data",
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "dim_reduced_data.png"]))

    ## IF asked to hypertune ##
    # from sklearn.mixture import GaussianMixture
    # aic_vals, bic_vals = [], []
    # n_components_range = range(1, 10)
    # for n in n_components_range:
    #     gmm = GaussianMixture(n_components=n, random_state=42)
    #     gmm.fit(reduced_dim_data)
    #     aic_vals.append(gmm.aic(reduced_dim_data))
    #     bic_vals.append(gmm.bic(reduced_dim_data))

    # # Plot AIC/BIC and save
    # pyplot.figure()
    # pyplot.plot(n_components_range, aic_vals, label="AIC")
    # pyplot.plot(n_components_range, bic_vals, label="BIC")
    # pyplot.xlabel("Number of Components")
    # pyplot.ylabel("Score")
    # pyplot.title("AIC/BIC to Choose Optimal GMM Components")
    # pyplot.legend()
    # pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "gmm_aic_bic_selection.png"]))

    
    # Step 4: Density estimation using GMM
    estimator = DensityEstimator(reduced_dim_data, dim_reducer, high_dim_feature_names=processed_data.columns)
    estimator.fit_distribution(n_components=4)
    predicted_clusters = estimator.gmm.predict(reduced_dim_data)

    # Step 5: Plot density estimation results
    plot_density_estimation_results(
        X=reduced_dim_data,
        Y_=predicted_clusters,
        means=estimator.gmm.means_,
        covariances=estimator.gmm.covariances_,
        title="GMM Density Estimation"
    )
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "density_estimation.png"]))

    # Step 6: Plot left vs right parties (using ideology column if needed)
    # Optional: Assuming you split based on PCA dimension
    left_mask = data_loader.party_data["lrgen"] < 5
    right_mask = data_loader.party_data["lrgen"] >= 5

    left = reduced_dim_data[left_mask]
    right = reduced_dim_data[right_mask]

    pyplot.figure()
    splot = pyplot.subplot()
    scatter_plot(left, color="blue", splot=splot, label="Left-wing")
    scatter_plot(right, color="red", splot=splot, label="Right-wing")
    pyplot.savefig(Path(__file__).parents[1].joinpath(*["plots", "left_right_parties.png"]))
    pyplot.title("Lefty/righty parties")


    # Step 7: Plot Finnish parties
    plot_finnish_parties(reduced_dim_data)

    print("Analysis Complete")
