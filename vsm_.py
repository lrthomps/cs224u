from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy
import scipy.spatial.distance
from scipy.stats import spearmanr
import torch
import utils

__author__ = "Christopher Potts"
__version__ = "CS224u, Stanford, Spring 2022"


def euclidean(u, v):
    return np.sqrt(np.sum((u-v)**2, axis=-1))


def vector_length(u):
    return np.sqrt(u.dot(u))


def length_norm(u):
    return u / vector_length(u)


def cosine(u, v):
    return 1 - np.sum(np.multiply(u, v), axis=-1) / np.sqrt(
    np.sum(u**2, axis=-1)*np.sum(v**2, axis=-1))


def correlation(u, v):
    return cosine(u - np.mean(u, axis=-1)[:, np.newaxis], 
                  v - np.mean(v, axis=-1)[:, np.newaxis])


def matching(u, v):
    return np.sum(np.minimum(u, v), axis=-1)


def jaccard(u, v):
    return 1.0 - matching(u, v) / np.sum(np.maximum(u, v), axis=-1)


def word_relatedness_evaluation(dataset_df, vsm_df, distfunc=cosine):
    """
    Main function for word relatedness evaluations used in the assignment
    and bakeoff. The function makes predictions for word pairs in
    `dataset_df` using `vsm_df` and `distfunc`, and it returns a copy of
    `dataset_df` with a new column `'prediction'`, as well as the Spearman
    rank correlation between those predictions and the `'score'` column
    in `dataset_df`.

    The prediction for a word pair (w1, w1) is determined by applying
    `distfunc` to the representations of w1 and w2 in `vsm_df`. We return
    the negative of this value since it is assumed that `distfunc` is a
    distance function and the scores in `dataset_df` are for positive
    relatedness.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Required to have columns {'word1', 'word2', 'score'}.

    vsm_df : pd.DataFrame
        The vector space model used to get representations for the
        words in `dataset_df`. The index must contain every word
        represented in `dataset_df`.

    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If any words in `dataset_df` are not in the index of `vsm_df`.

    Returns
    -------
    tuple (dataset_df, rho)
        Where `dataset_df` is a `pd.DataFrame` -- a copy of the
        input with a new column `'prediction'` -- and `rho` is a float
        giving the Spearman rank correlation between the `'score'`
        and `prediction` values.

    """
    dataset_df = dataset_df.copy()

    dataset_vocab = set(dataset_df.word1.values) | set(dataset_df.word2.values)

    vsm_vocab = set(vsm_df.index)

    missing = dataset_vocab - vsm_vocab

    if missing:
        raise ValueError(
            "The following words are in the evaluation dataset but not in the "
            "VSM. Please switch to a VSM with an appropriate vocabulary:\n"
            "{}".format(sorted(missing)))

    dataset_df['prediction'] = -distfunc(
        vsm_df.loc[dataset_df.word1].values, vsm_df.loc[dataset_df.word2].values)

    rho = None

    if 'score' in dataset_df.columns:
        rho, pvalue = spearmanr(
            dataset_df.score.values,
            dataset_df.prediction.values)

    return dataset_df, rho
