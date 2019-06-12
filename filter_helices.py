#!/usr/bin/env python
"""
File Name : filter_helices_not_intersecting_membrane.py
Creation Date : 12-06-2019
Last Modified : Mi 12 Jun 2019 18:34:07 CEST
Author : Luca Deininger
Function of the script :
"""

def get_membrane_intersecting_helices(helix_c_alphas, key, normal, middle):
    """
    Filters helix_c_alphas for one key not intersecting the membrane (currently without thickness).
    If C-alphas only on one site of membrane -> nontm
    If C-alphas on both site of membrane -> tm
    Site of membrane of C-alpha determined by termining the sign of the distance of C-alpha to membrane.

    """
    filt_helices = []
    for helix in helix_c_alphas[key]:
        distances = []
        for c_alpha in helix:

            dist = scalar_p(c_alpha, normal) - scalar_p(middle, normal)
            distances.append(dist)

        nr_above = len([x for x in distances if x >= 0])
        nr_below = len([x for x in distances if x < 0])

        
        # TM helix
        if nr_above > 0 and nr_below > 0:
            filt_helices.append(helix)
        # NON-TM helix
        elif nr_above > 0 or nr_below > 0:
            print("\n\n\nNon-TM helix found\n\n\n")
        else:
            print("no distances calculated")
            continue

    helix_c_alphas[key] = filt_helices
    return helix_c_alphas


def scalar_p(v1, v2):
    """
    Returns scalar product of two 3-dimensional vectors.
    """
    x = 0.0
    for i in range(3):
        x += float(v1[i]*v2[i])
    return x

