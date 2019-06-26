
import numpy as np
import matplotlib.pyplot as plt
import sys


def approximate_membrane(pdb_id, helices_c_alphas, helices_annotation):
    """
    approximates the membrane and returns normal
    :param pdb_id:
    :param helices_c_alphas:
    :param helices_annotation: annotations used to find tm matrices and changes annotations based on refinement
    :return: membrane_normal, membrane_position
    """

    tm_helices = get_tm_helices(helices_annotation[pdb_id], helices_c_alphas[pdb_id])
    if len(tm_helices) == 0:
        # if there is no tm helix return null vectors to represent no membrane:
        return [0, 0, 0], [0, 0, 0]

    axis = approximate_membrane_axis(tm_helices)
    position = approximate_membrane_position(tm_helices)
    if all(x == 0 for x in axis):
        print("Couldn't approximate axis:", pdb_id)
        return [0, 0, 0], [0, 0, 0]

    refine_annotation(helices_annotation[pdb_id], helices_c_alphas[pdb_id], axis, position)

    return axis, position


def refine_annotation(helices_annotation, helices_c_alphas, axis, position):
    """
    checks whether the helices intersect with the given membrane an changes the annotation according to it.
    :return: returns new annotations, boolean if something changed
    """
    plane = create_plane_equation(axis, position)
    changed = False
    for idx in range(len(helices_c_alphas)):

        tm_matrix = helix_in_membrane_cuts_membrane(helices_c_alphas[idx], plane)

        if tm_matrix and helices_annotation[idx][0] == 1 or not tm_matrix and helices_annotation[idx][0] == 0:
            pass
        else:
            changed = True
            if tm_matrix:
                helices_annotation[idx][0] = 1
            else:
                helices_annotation[idx][0] = 0

    return helices_annotation, changed


def helix_in_membrane_cuts_membrane(helix_c_alphas, plane):
    positive = False
    negative = False

    for c_alpha in helix_c_alphas:
        dist = get_distance_point_plane(c_alpha, plane)
        if dist < 0:
            negative = True
        else:
            positive = True

        if positive and negative:
            return True

    return False


def get_distance_point_plane(point, plane):
    p1 = plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3]
    p2 = np.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])

    dist = p1 / p2

    return dist


def create_plane_equation(membrane_normal, membrane_position):
    plane = list(membrane_normal)
    plane.append(-(sum([membrane_normal[i] * membrane_position[i] for i in range(len(membrane_normal))])))

    return plane


def approximate_membrane_position(helices):
    """
    approximate the membrane position by taking the average middle point of all tm-helices
    :param helices: helices list
    :return: the approximated middle point of the membrane
    """
    points = []

    for helix in helices:
        points.append(get_middle(helix))

    return get_middle(points)


def get_middle(helix):
    """
    returns the middle point of the given helix
    :param helix: list of ca atom position vectors
    :return: middle point of helix
    """
    if len(helix) == 0:
        return [0, 0, 0]  # TODO filter empty helices before they get to here

    if len(helix) % 2 == 1:
        return helix[int(len(helix)/2)]
    else:
        return mean_of_points([helix[int(len(helix)/2)], helix[int(len(helix)/2-1)]])


def approximate_membrane_axis(helices):
    """
    approximates the membrane normal vector by using single value decomposition over the calculated helix axis.
    """
    helix_axis = []
    for helix in helices:
        axis = approximate_helix_axis(helix)
        helix_axis.append(axis)

    if len(helix_axis) < 1:
        # print("skipping because too few helices")
        return [0, 0, 0]

    return fit_line(helix_axis)


def approximate_helix_axis(c_alphas):
    """
    approximates the helix axis by looking at the axis in every little part of the axis
    and then using single value decomposition to fit an axis through all the part-axis.
    :param c_alphas:
    :return:
    """

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
    """
    does single value decomposition on the given vector list.
    :param vector_list:
    :return:
    """
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


def get_tm_helices(helices_annotation, helices_c_alphas):
    indices_tm_helices = []
    for i in range(len(helices_annotation)):
        if helices_annotation[i][0] == 1:
            indices_tm_helices.append(i)

    tm_helices = []
    for idx in indices_tm_helices:
        tm_helices.append(helices_c_alphas[idx])

    return tm_helices
