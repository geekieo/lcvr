import numpy as np
import tensorflow as tf

from losses import TripletLoss._pairwise_distances as _pairwise_distances
# from losses import _get_anchor_positive_triplet_mask
# from losses import _get_anchor_negative_triplet_mask
# from losses import batch_hard_triplet_loss
# from losses import calculate_loss



def pairwise_distance_np(feature, squared=False):
    """Computes the pairwise distance matrix in numpy.
    Args:
        feature: 2-D numpy array of size [number of data, feature dimension]
        squared: Boolean. If true, output is the pairwise squared euclidean
                 distance matrix; else, output is the pairwise euclidean distance matrix.
    Returns:
        pairwise_distances: 2-D numpy array of size
                            [number of data, number of data].
    """
    triu = np.triu_indices(feature.shape[0], 1)
    upper_tri_pdists = np.linalg.norm(feature[triu[1]] - feature[triu[0]], axis=1)
    if squared:
        upper_tri_pdists **= 2.
    num_data = feature.shape[0]
    pairwise_distances = np.zeros((num_data, num_data))
    pairwise_distances[np.triu_indices(num_data, 1)] = upper_tri_pdists
    # Make symmetrical.
    pairwise_distances = pairwise_distances + pairwise_distances.T - np.diag(
            pairwise_distances.diagonal())
    return pairwise_distances


def test_pairwise_distances():
    """Test the pairwise distances function."""
    num_data = 64
    feat_dim = 6

    embeddings = np.random.randn(num_data, feat_dim).astype(np.float32)
    embeddings[1] = embeddings[0]  # to get distance 0

    with tf.Session() as sess:
        for squared in [True, False]:
            res_np = pairwise_distance_np(embeddings, squared=squared)
            res_tf = sess.run(_pairwise_distances(embeddings, squared=squared))
            assert np.allclose(res_np, res_tf)