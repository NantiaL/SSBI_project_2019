
import numpy as np


def approximate_membrane_position(helices, pdb_id):
    points = []

    for helix in helices[pdb_id]:
        points.append(get_middle(helix))

    return get_middle(points)


def get_middle(helix):
    if len(helix) == 0:
        return [0, 0, 0]  # TODO filter empty helices before they get to here

    if len(helix) % 2 == 1:
        return helix[int(len(helix)/2)]
    else:
        return mean_of_points([helix[int(len(helix)/2)], helix[int(len(helix)/2-1)]])


def approximate_membrane_axis(helices, pdb_id):
    helix_axis = []
    for helix in helices[pdb_id]:
        axis = approximate_helix_axis(helix)
        helix_axis.append(axis)

    if len(helix_axis) < 1:
        print("skipping because too few helices:", pdb_id)
        return [0, 0, 0]

    return fit_line(helix_axis)


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

def mean_of_points(points):
    sum_vector = list(points[0])

    for i in range(1, len(points)):
        for j in range(len(list(points[i]))):
            sum_vector[j] += points[i][j]

    for i in range(len(sum_vector)):
        sum_vector[i] = sum_vector[i]/len(points)

    return sum_vector


def calculate_xml_normal_to_base_coordninates(pdbtm_m, pdb_id):

    vectors = extract_tmatrix_vectors(pdbtm_m, pdb_id)
    normal = extract_normal_vector(pdbtm_m, pdb_id)

    parts = []
    for i in range(len(normal)):
        parts.append([x*normal[i] for x in vectors[i]])

    base_normal = parts[0]

    for i in range(1, len(parts)):
        for j in range(len(parts[i])):
            base_normal[j] += parts[i][j]
    return normalize_vector(base_normal)


def extract_tmatrix_vectors(pdbtm_m, pdb_id):
    vectors = []

    for i in range(1, 4):
        vector = []
        for j in range(1, 4):
            vector.append(float(pdbtm_m[pdb_id][i][j]))
        vectors.append(vector)

    return vectors


def extract_normal_vector(pdbtm_m, pdb_id):
    normal = []

    for i in range(1, 4):
        normal.append(float(pdbtm_m[pdb_id][0][i]))

    return normal
