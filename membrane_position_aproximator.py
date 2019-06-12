
import numpy as np


def approximate_helix_axis(c_alphas):

    normal_vectors = []
    for i in range(1, len(c_alphas)-2):

        bisector_one = get_bisector(c_alphas, i)
        bisector_two = get_bisector(c_alphas, i+1)

        normal_vectors.append(np.cross(bisector_one, bisector_two))

    if len(normal_vectors) == 0:
        return [0, 0, 0]

    return fit_line(normal_vectors)


def get_bisector(c_alphas, index):
    vector_to_prev_ca = c_alphas[index - 1] - c_alphas[index]
    vector_to_next_ca = c_alphas[index + 1] - c_alphas[index]

    return normalize_vector(vector_to_prev_ca + vector_to_next_ca)


def normalize_vector(vector):
    magnitude = np.sqrt(sum([x*x for x in vector]))

    normalized_vector = []
    for x in vector:
        normalized_vector.append(x/magnitude)

    return normalized_vector


def fit_line(vector_list):
    data = np.array(vector_list)
    uu, dd, vv = np.linalg.svd(data)

    return vv[0]








