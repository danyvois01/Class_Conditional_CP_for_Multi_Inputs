# Class conditional conformal prediction for multiple inputs by p-value aggregation

This repository contains the code necessary to reproduce the experimental results presented in the paper *Class conditional conformal prediction for multiple inputs by p-value aggregation* of [Jean-Baptiste Fermanian](https://jeanbaptistefermanian.github.io/), [Mohamed Hebiri](https://perso.math.u-pem.fr/hebiri.mohamed/) and [Joseph Salmon](https://josephsalmon.eu) ([arXiv](url)).

## General

-   `utils.py`: Contains all functions used for constructing prediction sets and generating synthetic data.

To accelerate computation, quantiles for certain distributions are precomputed:

-   `Bin_quantile_lvl0.1_m100`: Contains quantiles of the Binomial distribution B(m, \alpha) for m = 1, ..., 100 and \alpha = 0.1 (values used in the paper).
-   `nHgeo_quantile_lvl0.1_m100n1000`: Contains quantiles of the Beta-Binomial distribution BetaBinomial(m, n+1-k, k) for m = 1, ..., 100 ;  n = 1, ..., 1000; and  k = \lfloor(n+1)\alpha\rfloor.

## Plot envelope

-   `viz_envelope.py`: generates Figure 1.

## Synthetic Data Experiments

-   `CP_synthetic.py`: Runs all experiments on synthetic data and generates the corresponding figures (Figures 2, 4, and 5 in the paper).

## Real Data Experiments

### Preprocessing

The `Pre-processing PlantClef` directory contains all necessary functions to compute predicted scores for each image before applying conformal prediction methods.

-   **Dataset**: We use data from LifeClef 2015 ([LifeClef 2015 Plant Task](https://www.imageclef.org/lifeclef/2015/plant)), consisting of two datasets that need to be downloaded:

    -   `PlantCLEF2015TrainingData`
    -   `PlantCLEF2015TestDataWithAnnotations`

You can automatically download these datasets using the script `download_plantclef.py`. This script downloads the datasets from the Pl@ntNet server and extracts them into a specified directory.

-   **Data Split**:

    -   `Newsplit.py`: Splits the merged dataset into three equally sized parts, preserving observation structure (i.e., images from the same observation remain in the same split). Only the indices of the new sets are returned.

-   **Model**: We fine-tune the pre-trained model `resnet50_weights_best_acc.tar` provided by Garcin et al. (2021) ([PlantNet-300K repository](https://github.com/plantnet/PlantNet-300K/)) which must also be downloaded. The file `utils_PlantNet300K.py` comes from this source.

-   **Training**:

    -   `model_training.py`: Trains the models using the new split. It relies on `utils_PlantNet300K.py` (from the [PlantNet-300K repository](https://github.com/plantnet/PlantNet-300K/)) to properly load the pre-trained model.

-   **Score Computation**:

    -   `scores_computation.py`: Computes the scores (output of the classifier for each class) for both the test and calibration datasets.
    -   `model.pth` contained a pretrained model which can directly be used for the computation of the scores.

### Conformal Prediction

-   `CP_Truedata.py`: Computes coverage (both marginal and conditional) and average size (both marginal and conditional) of the different methods on real data, across various splits.

-   `plots_truedata.py`: Generates Figures 3, 7, and 8 related to real data. The necessary data files are provided:

    -   `Plantnet_shuffle_temp1alpha0.1minclasssize20.npy`
    -   `Plantnet_shuffle_temp20alpha0.1minclasssize20.npy`
    -   `Plantnet_trueobs_temp1alpha0.1minclasssize20.npy`
