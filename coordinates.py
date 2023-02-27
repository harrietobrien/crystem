# https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography


import numpy as np


def traditionalTransformation():
    # cell parameters
    a = 7.55
    b = 4.99
    c = 12.50
    # degrees to radians
    alpha = 122.5 * np.pi / 180
    beta = (95 + 18 / 60) * np.pi / 180
    gama = (118 + 54 / 60) * np.pi / 180
    # x/a, y/a, z/a
    V1 = np.array([-0.2812, -0.0628, 0.1928])
    V2 = np.array([-0.2308, -0.0972, 0.2931])
    V3 = np.array([-0.3639, -0.1913, 0.3521])

    n2 = (np.cos(alpha) - np.cos(gama) * np.cos(beta)) / np.sin(gama)
    M = np.array([[a, 0, 0], [b * np.cos(gama), b * np.sin(gama), 0],
                  [c * np.cos(beta), c * n2, c * np.sqrt(np.sin(beta) ** 2 - n2 ** 2)]])
    # row x matrix
    dcm1 = (V1 - V2) @ M
    # sqrt(dot product)
    L12 = np.sqrt(dcm1 @ dcm1)
    return


def latticeVectorTransformation():
    # Pick your lattice vectors
    a1 = np.array([...])
    a2 = np.array([...])
    a3 = np.array([...])
    # Build matrix of lattice vectors stored column-wise
    A = np.vstack([a1, a2, a3]).T
    # and get its inverse
    A_inv = np.linalg.inv(A)
    # Set of fractional coordinates stored row-wise
    Y = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    # Compute the cartesian coordinates
    X = np.matmul(A, Y.T).T
    # Perform the inverse operation to get fractional coordinates
    Y_ = np.matmul(A_inv, X.T).T
    return
