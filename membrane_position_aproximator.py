
import numpy as np

def approximate_helix_axis(c_alphas):

    normal_vectors = []
    for i in range(1, len(c_alphas)-2):

        bisector_one = get_bisector(c_alphas, i)
        bisector_two = get_bisector(c_alphas, i+1)

        normal_vectors.append(np.cross(bisector_one, bisector_two))

    if len(normal_vectors) == 0:
        return [0, 0, 0]
    # print(normal_vectors)
    sum_vector = [0, 0, 0]
    for x in normal_vectors:
        for i in range(len(x)):
            sum_vector[i] += x[i]
    mean_vector = normalize_vector([x/len(normal_vectors) for x in sum_vector])
    # print("mean:", mean_vector)

    return mean_vector


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












