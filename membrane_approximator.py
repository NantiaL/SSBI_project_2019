
import numpy as np
import matplotlib.pyplot as plt
import sys


global_min_distances = []

def approximate_membrane(pdb_id, helices_c_alphas, helices_annotation):

    times = 0
    while True:
        tm_helices = get_tm_helices(helices_annotation[pdb_id], helices_c_alphas[pdb_id])
        axis = approximate_membrane_axis(tm_helices)
        position = approximate_membrane_position(tm_helices)
        if all(x == 0 for x in axis):
            print("Couldn't approximate axis:", pdb_id)
            break

        changed = False
        new_helices_anno, changed = refine_annotation(helices_annotation[pdb_id], helices_c_alphas[pdb_id], axis, position)

        times += 1
        if not changed or times > 5:
            #print("Stopped:", times)
            break

    return axis, position


def refine_annotation(helices_annotation, helices_c_alphas, axis, position):
    assumed_thickness = 0.7

    plane = create_plane_equation(axis, position)
    changed = False
    for idx in range(len(helices_c_alphas)):
        # print()
        # print(helices_annotation[idx])
        #tm_matrix = helix_in_membrane_thickness(helices_c_alphas[idx], plane, assumed_thickness)
        tm_matrix = helix_in_membrane_cuts_membrane(helices_c_alphas[idx], plane)

        if tm_matrix and helices_annotation[idx][0] == 1 or not tm_matrix and helices_annotation[idx][0] == 0:
            pass
        else:
            changed = True
            # print("changed Annotation!")
            if tm_matrix:
                helices_annotation[idx][0] = 1
            else:
                helices_annotation[idx][0] = 0
            #print(helices_annotation[idx])
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


def helix_in_membrane_thickness(helix_c_alphas, plane, thickness):
    distances = []
    for c_alpha in helix_c_alphas:
        dist = get_distance_point_plane(c_alpha, plane)
        distances.append(dist)

    min_dist = np.abs(min(distances))
    #print(min_dist)
    if min_dist < thickness:
        return True

    return False

def approximate_membrane_thickness(helices, pdb_id, membrane_normal, membrane_position):
    """
    approximates the thickness of the membrane by looking at the smallest distance to an end of a helix
    on each side off the membrane.

    TODO: DOESNT WORK RETURNS COMPLETELY WRONG APPROXIMATIONS!
    """

    plane = create_plane_equation(membrane_normal, membrane_position)
    distances = []

    for helix in helices[pdb_id]:
        if len(helix) == 0:
            continue
        dist_start, dist_end = get_distances_to_plane(helix, plane)

        distances.append(dist_start)
        distances.append(dist_end)

    biggest_negative = -sys.maxsize - 1
    smallest_positive = sys.maxsize

    for dist in distances:
        if biggest_negative < dist < 0:
            biggest_negative = dist
        if 0 < dist < smallest_positive:
            smallest_positive = dist

    # return round((smallest_positive - biggest_negative) * 100, 2)
    return 0


def get_distances_to_plane(helix, plane):

    start_dist = get_distance_point_plane(helix[0], plane)
    end_dist = get_distance_point_plane(helix[-1], plane)

    return start_dist, end_dist


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
    :param helices: all helices dict
    :param pdb_id: id of the file to look at
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
    :param helices:
    :param pdb_id:
    :return:
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

"""
def calculate_xml_membrane_position(pdbtm_m, pdb_id):
    vectors = extract_tmatrix_vectors_with_T(pdbtm_m, pdb_id)

    for vector in vectors:
        for i in range(len(vector)-1):
            vector[i] = vector[i] * vector[3]

    sum_vector = []
    for i in range(3):
        sum_coordinate = 0
        for vector in vectors:
            sum_coordinate += vector[i]

        sum_vector.append(sum_coordinate * -1)
    return sum_vector


def calculate_xml_normal_to_base_coordinates(pdbtm_m, pdb_id):

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


def get_xml_membrane_thickness(pdbtm_m, key):
    normal = extract_normal_vector(pdbtm_m, key)

    return normal[2]*2


def extract_tmatrix_vectors(pdbtm_m, pdb_id):
    vectors = []

    for i in range(1, 4):
        vector = []
        for j in range(1, 4):
            vector.append(float(pdbtm_m[pdb_id][i][j]))
        vectors.append(vector)

    return vectors


def extract_tmatrix_vectors_with_T(pdbtm_m, pdb_id):
    vectors = []

    for i in range(1, 4):
        vector = []
        for j in range(1, 5):
            vector.append(float(pdbtm_m[pdb_id][i][j]))
        vectors.append(vector)

    return vectors


def extract_normal_vector(pdbtm_m, pdb_id):
    normal = []

    for i in range(1, 4):
        normal.append(float(pdbtm_m[pdb_id][0][i]))

    return normal 
"""