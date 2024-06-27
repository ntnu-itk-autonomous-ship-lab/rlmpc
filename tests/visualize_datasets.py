"""
This script is used to visualize the datasets used for training the tracking VAE model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rlmpc.common.datasets as rl_ds
import seaborn as sns

if __name__ == "__main__":
    batch_size = 32
    data_dir = Path.home() / "Desktop/machine_learning/data/tracking_vae/"
    training_data_npy_filename = "tracking_vae_training_data_rogaland1.npy"
    test_data_npy_filename = "tracking_vae_test_data_rogaland1.npy"
    training_data_npy_filename2 = "tracking_vae_training_data_rogaland2.npy"
    test_data_npy_filename2 = "tracking_vae_test_data_rogaland2.npy"

    training_dataset1 = rl_ds.TrackingObservationDataset(training_data_npy_filename, data_dir).get_data()
    test_dataset1 = rl_ds.TrackingObservationDataset(test_data_npy_filename, data_dir).get_data()
    training_dataset2 = rl_ds.TrackingObservationDataset(training_data_npy_filename2, data_dir).get_data()
    test_dataset2 = rl_ds.TrackingObservationDataset(test_data_npy_filename2, data_dir).get_data()

    titles = ["Rel dist", "Rel speed x", "Rel speed y", "Var speed x", "Var speed y", "Cov speed xy"]

    df_training1 = pd.DataFrame(training_dataset1[::20, :, 1:3, :].reshape(-1, 2), columns=["speed x", "speed y"])
    df_test1 = pd.DataFrame(test_dataset1[::20, :, 1:3, :].reshape(-1, 2), columns=["speed x", "speed y"])

    # Create dataframes for dataset 2
    df_training2 = pd.DataFrame(training_dataset2[::20, :, 1:3, :].reshape(-1, 2), columns=["speed x", "speed y"])
    df_test2 = pd.DataFrame(test_dataset2[::20, :, 1:3, :].reshape(-1, 2), columns=["speed x", "speed y"])

    # find indices where both speed x and speed y are 0:
    td1sub = training_dataset1[::10, :, :, :].reshape(-1, 6, 10)
    idxs = np.where((td1sub[:, 1, :] == 0) & (td1sub[:, 2, :] == 0))
    td1_0speed = td1sub[idxs[0], :, idxs[1]].reshape(-1, 6, 10)

    df_training1["dataset"] = "training_dataset1"
    df_test1["dataset"] = "test_dataset1"
    df_training2["dataset"] = "training_dataset2"
    df_test2["dataset"] = "test_dataset2"

    # Concatenate all dataframes into one
    combined_df = pd.concat([df_training1, df_test1, df_training2, df_test2], ignore_index=True)

    g = sns.JointGrid(
        data=combined_df,
        x="speed x",
        y="speed y",
        hue="dataset",
        xlim=(-1.0, 1.0),
        ylim=(-1.0, 1.0),
    )
    g.plot_joint(sns.scatterplot, s=100, alpha=0.5)
    g.plot_marginals(sns.stripplot, alpha=0.5, hue="dataset", dodge=True)
    # g.plot_marginals(sns.kdeplot, fill=False, clip=(-1, 1), bw_adjust=0.2)

    # create seaborn grid for the below plots
    fig, axs = plt.subplots(4, 1, figsize=(15, 10))
    plt.show(block=False)
    for i in range(2, 6):
        ax = axs[i - 2]
        if i == 2:
            training_samples_i = training_dataset1[::10, :, 0, :].flatten()
            test_samples_i = test_dataset1[::10, :, 0, :].flatten()
            training_samples_i2 = training_dataset2[::10, :, 0, :].flatten()
            test_samples_i2 = test_dataset2[::10, :, 0, :].flatten()
        else:
            training_samples_i = training_dataset1[::10, :, i, :].flatten()
            test_samples_i = test_dataset1[::10, :, i, :].flatten()
            training_samples_i2 = training_dataset2[::10, :, i, :].flatten()
            test_samples_i2 = test_dataset2[::10, :, i, :].flatten()

        training_samples_i = training_samples_i[np.where(training_samples_i < 0.99)]
        test_samples_i = test_samples_i[np.where(test_samples_i < 0.99)]
        training_samples_i2 = training_samples_i2[np.where(training_samples_i2 < 0.99)]
        test_samples_i2 = test_samples_i2[np.where(test_samples_i2 < 0.99)]

        sns.kdeplot(
            training_samples_i.flatten(),
            ax=ax,
            label="Training1",
            legend=True,
            fill=True,
            clip=(-1, 1),
            bw_adjust=0.2,
            warn_singular=False,
        )
        sns.kdeplot(
            test_samples_i.flatten(),
            ax=ax,
            label="Test1",
            legend=True,
            fill=True,
            clip=(-1, 1),
            bw_adjust=0.2,
            warn_singular=False,
        )
        sns.kdeplot(
            training_samples_i2.flatten(),
            ax=ax,
            label="Training2",
            legend=True,
            fill=True,
            clip=(-1, 1),
            bw_adjust=0.2,
            warn_singular=False,
        )
        sns.kdeplot(
            test_samples_i2.flatten(),
            ax=ax,
            label="Test2",
            legend=True,
            fill=True,
            clip=(-1, 1),
            bw_adjust=0.2,
            warn_singular=False,
        )
        ax.legend()

    print("Done")
