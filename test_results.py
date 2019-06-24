
import os
import pickle
import requests
import math
import numpy as np
import xml.etree.cElementTree as ET

pdbtm_ids_path = "serialized/result_test_serialized_pdbtm_ids.dic"


def test_result_against_pdbtm(pdbid, helix_annotations, membrane_axis, membrane_position):
    """
    tests the results against the pdbtm database.
    returns tuple (correctly_classified_globular/tm, angle_between_approx_and_real, distance_between_approx_and_real, false_positive)
    :param pdbid:
    :param helix_annotations:
    :param membrane_axis:
    :param membrane_position:
    :return:
    """
    pdbtm_ids = load_pdbtm_ids()

    if pdbid in pdbtm_ids:
        xml_file = download_xmlfile(pdbid)
        if "404 Not Found" in xml_file:
            # print("File not found!")
            raise IOError("Couldn't download file from database!")
        else:

            if all(v == 0 for v in membrane_axis):
                # print("Wrongly classified as globular or axis calculation not possible!")
                return False, -1, -1, False
            else:
                xml_membrane, xml_ss = parse_xmlfile_string(xml_file)
                xml_mem_pos = get_xml_membrane_position(xml_membrane)
                xml_axis = get_xml_normal_to_base_coordinates(xml_membrane)
                angle = angle_between_vectors(membrane_axis, xml_axis)
                xml_plane_equation = create_plane_equation(xml_axis, xml_mem_pos)
                distance = get_distance_point_plane(membrane_position, xml_plane_equation)

                # print of all infos:
                # print_infos(pdbid, membrane_axis, xml_axis, angle, membrane_position, xml_mem_pos, distance)
                return True, angle, distance, False

    else:
        # print("Not in pdbtm. Should be globular")
        if all(v == 0 for v in membrane_axis):
            # print("Correct classified as globular.")
            return True, -1, -1, False
        else:
            # print("Wrongly classified as tm protein.")
            return False, -1, -1, True


def print_infos(pdbid, membrane_axis, xml_axis, angle, membrane_position, xml_mem_pos, distance):
    print("pdb id:", pdbid)
    print("Approximated normal:", membrane_axis)
    print("XML-Membrane axis :", xml_axis)
    print("Angle between axis:", angle)
    print("Approximated pos   :", membrane_position)
    print("XML-Membrane pos  :", xml_mem_pos)
    print("Distance to xml_plane:", distance)
    print()

def load_pdbtm_ids():
    if os.path.isfile(pdbtm_ids_path):
        return pickle.load(open(pdbtm_ids_path, "rb"))
    else:
        filepath = "pdbtm_all_list.txt"
        pdbtm_ids = {}
        for line in open(filepath, "r"):
            line = line.strip()
            pdbtm_id = line[0:4]
            if pdbtm_id in pdbtm_ids:
                continue
            else:
                pdbtm_ids[pdbtm_id] = 1

        pickle.dump(pdbtm_ids, open(pdbtm_ids_path, "wb"))

        return pdbtm_ids


def download_xmlfile(pdbid):
    url = "http://pdbtm.enzim.hu/data/database/" + pdbid[1:3] + "/" + pdbid + ".xml"
    xml_data = requests.get(url)

    file = ""
    for data in xml_data:
        file += data.decode('utf-8')

    return file


def parse_xmlfile_string(str_file):
    root = ET.fromstring(str_file)
    membrane = []
    secondary_structures = []
    for child in root:
        tag = child.tag[23:]
        if tag == "MEMBRANE":
            for child2 in child:
                tag_membrane = child2.tag[23:]
                if tag_membrane == "NORMAL":
                    x = child2.attrib["X"]
                    y = child2.attrib["Y"]
                    z = child2.attrib["Z"]
                    membrane = [("NORMAL", x, y, z)]

                if tag_membrane == "TMATRIX":
                    for row in child2:
                        membrane.append(
                            (row.tag[23:], row.attrib["X"], row.attrib["Y"], row.attrib["Z"], row.attrib["T"]))
        elif tag == "CHAIN":
            chainid = child.attrib["CHAINID"]
            for child2 in child:
                tag2 = child2.tag[23:]
                # extracting sequence
                if tag2 == "SEQ":
                    seq = child2.text
                    seq = seq.replace("\n", "")
                    seq = seq.replace(" ", "")
                    # sequence could be also included but not necessary at the moment
                    # pdbtms[pdbid].append(seq)
                # extracting sec structures
                elif tag2 == "REGION":
                    seq_beg = int(child2.attrib["seq_beg"])
                    seq_end = int(child2.attrib["seq_end"])
                    pdb_beg = int(child2.attrib["pdb_beg"])
                    pdb_end = int(child2.attrib["pdb_end"])
                    type_ = child2.attrib["type"]
                    # TODO: indices correct?
                    secondary_structures.append(
                        (chainid, seq[seq_beg - 1:seq_end], seq_beg, seq_end, pdb_beg, pdb_end, type_))

    # print(membrane)
    # print(secondary_structures)
    return membrane, secondary_structures


def get_distance_point_plane(point, plane):
    p1 = plane[0]*point[0] + plane[1]*point[1] + plane[2]*point[2] + plane[3]
    p2 = np.sqrt(point[0]*point[0] + point[1]*point[1] + point[2]*point[2])

    dist = p1 / p2

    return dist


def create_plane_equation(membrane_normal, membrane_position):
    plane = list(membrane_normal)
    plane.append(-(sum([membrane_normal[i] * membrane_position[i] for i in range(len(membrane_normal))])))

    return plane


def get_xml_membrane_position(xml_membrane):
    vectors = get_tmatrix_vectors_with_t(xml_membrane)

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


def get_xml_membrane_thickness(xml_membrane):
    normal = extract_normal_vector(xml_membrane)

    return normal[2]*2


def get_tmatrix_vectors(xml_membrane):
    vectors = []

    for i in range(1, 4):
        vector = []
        for j in range(1, 4):
            vector.append(float(xml_membrane[i][j]))
        vectors.append(vector)

    return vectors


def get_tmatrix_vectors_with_t(xml_membrane):
    vectors = []

    for i in range(1, 4):
        vector = []
        for j in range(1, 5):
            vector.append(float(xml_membrane[i][j]))
        vectors.append(vector)

    return vectors


def get_normal_vector(xml_membrane):
    normal = []

    for i in range(1, 4):
        normal.append(float(xml_membrane[0][i]))

    return normal


def normalize_vector(vector):
    magnitude = np.sqrt(sum([x*x for x in vector]))

    normalized_vector = []
    for x in vector:
        normalized_vector.append(x/magnitude)

    return normalized_vector


def get_xml_normal_to_base_coordinates(xml_membrane):

    vectors = get_tmatrix_vectors(xml_membrane)
    normal = get_normal_vector(xml_membrane)

    parts = []
    for i in range(len(normal)):
        parts.append([x*normal[i] for x in vectors[i]])

    base_normal = parts[0]

    for i in range(1, len(parts)):
        for j in range(len(parts[i])):
            base_normal[j] += parts[i][j]
    return normalize_vector(base_normal)


def dist_of_vectors(vector1, vector2):
    vector1 = list(vector1)
    vector2 = list(vector2)
    diff = []
    for i in range(len(vector1)):
        diff.append(vector1[i] - vector2[i])

    magnitude = 0

    for i in range(len(diff)):
        magnitude += diff[i]*diff[i]

    return np.sqrt(magnitude)


def angle_between_vectors(vector1, vector2):
    vector1_n = normalize_vector(vector1)
    vector2_n = normalize_vector(vector2)

    angle = math.degrees(np.arccos(np.clip(np.dot(vector1_n, vector2_n), -1.0, 1.0)))

    return angle


def run_tests():
    parse_xmlfile_string(download_xmlfile("1zll"))


if __name__ == "__main__":
    run_tests()