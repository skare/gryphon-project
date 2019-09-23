# This file is part of Gryphon.
#
# The constants in the Butcher tableaux below were discovered by
# Anne Kv\{ae}rn\{o} working at the Norwegian University of Science and Technology.

# More information regarding the ESDIRK methods can be found in her article:
# A. Kv\{ae}rn\{o}, Singly diagonally implicit Runge-Kutta methods with an explicit first stage, BIT Numerical Mathematics 44 (2004), no. 3, 489-502.

import numpy as np


def getTable(method):
    B = {}

    B['ESDIRK43a'] = {
        'gamma': 0.5728160625,
        'order': 4,  # Order of method
        'advSt': 4,  # Stage used to advance method (zero indexed)
        'tableau': np.array([[0, 0, 0, 0, 0],
                             [0.572816062500000, 0.572816062500000, 0, 0, 0],
                             [0.167235462041900, -0.142946536861288, 0.572816062500000, 0, 0],
                             [0.262603290273974, -0.311904327414786, 0.476484974640808, 0.572816062500000, 0],
                             [0.197216548321029, 0.176843783906613, 0.815442181403551, -0.762318576131192,
                              0.572816062500000]],
                            np.dtype('d'))
    }

    B['ESDIRK43b'] = {
        'gamma': 0.4358665215,
        'order': 3,  # Order of method
        'advSt': 3,  # Stage used to advance method (zero indexed)
        'tableau': np.array([[0, 0, 0, 0, 0],
                             [0.435866521500000, 0.435866521500000, 0, 0, 0],
                             [0.140737774731968, -0.108365551378832, 0.435866521500000, 0, 0],
                             [0.102399400616089, -0.376878452267324, 0.838612530151233, 0.435866521500000, 0],
                             [0.157024897860995, 0.117330441357768, 0.616678030391680, -0.326899891110444,
                              0.435866521500000]],
                            np.dtype('d'))
    }

    # Butcher tableau for ESDIRK 3/2
    B['ESDIRK32a'] = {
        'gamma': 0.4358665215,
        'order': 3,  # Order of method
        'advSt': 3,  # Stage used to advance method (zero indexed)
        'tableau': np.array([[0, 0, 0, 0],
                             [0.435866521500000, 0.435866521500000, 0, 0],
                             [0.490563388419108, 0.073570090080892, 0.435866521500000, 0],
                             [0.308809969973036, 1.490563388254108, -1.235239879727145, 0.435866521500000]],
                            np.dtype('d'))
    }

    B['ESDIRK32b'] = {
        'gamma': 0.2928932188,
        'order': 2,  # Order of method
        'advSt': 2,  # Stage used to advance method (zero indexed)
        'tableau': np.array([[0, 0, 0, 0],
                             [0.292893218800000, 0.292893218800000, 0, 0],
                             [0.353553390567523, 0.353553390632477, 0.292893218800000, 0],
                             [0.215482203122508, 0.686886723913539, -0.195262145836047, 0.292893218800000]],
                            np.dtype('d'))
    }

    return B[method]
