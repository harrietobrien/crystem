#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 16:30:20 2018

@author: harrietobrien
"""

from spglib_database import spglib_database
import numpy as np
import random


class get_spglib_database(object):

    def __init__(self, space_group_number):
        self.spacegroup_to_hall_number = \
            [1, 2, 3, 6, 9, 18, 21, 30, 39, 57,
             60, 63, 72, 81, 90, 108, 109, 112, 115, 116,
             119, 122, 123, 124, 125, 128, 134, 137, 143, 149,
             155, 161, 164, 170, 173, 176, 182, 185, 191, 197,
             203, 209, 212, 215, 218, 221, 227, 228, 230, 233,
             239, 245, 251, 257, 263, 266, 269, 275, 278, 284,
             290, 292, 298, 304, 310, 313, 316, 322, 334, 335,
             337, 338, 341, 343, 349, 350, 351, 352, 353, 354,
             355, 356, 357, 358, 359, 361, 363, 364, 366, 367,
             368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
             378, 379, 380, 381, 382, 383, 384, 385, 386, 387,
             388, 389, 390, 391, 392, 393, 394, 395, 396, 397,
             398, 399, 400, 401, 402, 404, 406, 407, 408, 410,
             412, 413, 414, 416, 418, 419, 420, 422, 424, 425,
             426, 428, 430, 431, 432, 433, 435, 436, 438, 439,
             440, 441, 442, 443, 444, 446, 447, 448, 449, 450,
             452, 454, 455, 456, 457, 458, 460, 462, 463, 464,
             465, 466, 467, 468, 469, 470, 471, 472, 473, 474,
             475, 476, 477, 478, 479, 480, 481, 482, 483, 484,
             485, 486, 487, 488, 489, 490, 491, 492, 493, 494,
             495, 497, 498, 500, 501, 502, 503, 504, 505, 506,
             507, 508, 509, 510, 511, 512, 513, 514, 515, 516,
             517, 518, 520, 521, 523, 524, 525, 527, 529, 530]
        self.space_group_number = space_group_number
        self.hall_number = self.get_hall_number()
        self.space_group = spglib_database[self.hall_number]['Space Group']
        self.multiplicity = spglib_database[self.hall_number]['Multiplicity']
        self.site_symmetries = spglib_database[self.hall_number]['Site Symmetry']
        self.coordinates = spglib_database[self.hall_number]['Coordinates']
        self.unique_site_symmetries = self.get_uss()
        self.wtran = self.get_wtran()
        self.btran = self.get_btran()
        self.wop = self.get_wop()
        self.mlist = self.wyckoff_preparation_trace()

    def get_hall_number(self):
        return str(self.spacegroup_to_hall_number[self.space_group_number - 1])

    # translation of the Wyckoff positions
    def get_wtran(self):
        translation_vectors = []
        for mult in self.coordinates:
            tmp_translation_vectors = []
            for vector in mult:
                vector = vector.replace('(', '').replace(')', '')
                coordinate = np.array([])
                for num in vector.split(','):
                    if all((char.isalpha() or char == '-') for char in num):
                        coordinate = np.append(coordinate, 0)
                    elif any(char == '+' for char in num):
                        for char in num:
                            if char.isalpha() or char == '+' or char == '-':
                                num = num.replace(char, '')
                        coordinate = np.append(coordinate, eval(num))
                    elif all(char.isdigit() or char == '/' for char in num):
                        coordinate = np.append(coordinate, eval(num))
                tmp_translation_vectors.append(coordinate)
            translation_vectors.append(tmp_translation_vectors)
        return translation_vectors

    # centering operations for the Bravais lattice type
    def get_btran(self):
        for space_group in self.space_group:
            if space_group[0] == 'P':
                return [[0, 0, 0]]
            elif space_group[0] == 'I':
                return [[0, 0, 0], [0.5, 0.5, 0.5]]
            elif space_group[0] == 'C':
                return [[0, 0, 0], [0.5, 0.5, 0]]
            elif space_group[0] == 'F':
                return [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
            elif space_group[0] == 'A':
                return [[0, 0, 0], [0, 0.5, 0.5]]

    # constraints on Wyckoff positions
    def get_wop(self):
        constraint_matrices = []
        for mult in self.coordinates:
            tmp_constraint_matrices = []
            for vector in mult:
                vector = vector.replace('(', '').replace(')', '')
                matrix = []
                for num in vector.split(','):
                    if all(char.isdigit() or char == '/' for char in num):
                        matrix.append(np.array([0, 0, 0]))
                    else:
                        for char in num:
                            if not (char.isalpha() or char == '-'):
                                num = num.replace(char, '')
                        if num == 'x':
                            matrix.append(np.array([1, 0, 0]))
                        elif num == '-x':
                            matrix.append(np.array([-1, 0, 0]))
                        elif num == 'y':
                            matrix.append(np.array([0, 1, 0]))
                        elif num == '-y':
                            matrix.append(np.array([0, -1, 0]))
                        elif num == 'z':
                            matrix.append(np.array([0, 0, 1]))
                        elif num == '-z':
                            matrix.append(np.array([0, 0, -1]))
                tmp_constraint_matrices.append(np.vstack([matrix]))
            constraint_matrices.append(tmp_constraint_matrices)
        return constraint_matrices

    # multiplicity of Wyckoff positions
    def get_wmult(self):
        return self.multiplicity

    # unique site symmetries
    def get_uss(self):
        unique_sites = []
        for site in self.site_symmetries:
            if site not in unique_sites:
                unique_sites.append(site)
        return unique_sites

    # number of unique site symmetry groups
    def get_nsg(self):
        return len(self.unique_site_symmetries)

    # number of Wyckoff positions for each unique site symmetry
    def get_swycn(self):
        number_wyckoff_positions = []
        for site in self.unique_site_symmetries:
            number_wyckoff_positions.append(self.site_symmetries.count(site))
        return number_wyckoff_positions

    # multiplicity of Wyckoff positions for each unique site symmetry
    def swyc_mult(self):
        wyckoff_position_multiplicity = []
        for site in self.unique_site_symmetries:
            wyckoff_position_multiplicity.append(self.multiplicity
                                                 [self.site_symmetries.index(site)])
        return wyckoff_position_multiplicity

    def wyckoff_preparation_trace(self):
        mlist = []
        for mult in self.wop:
            mlist.append(random.choice(mult))
        return mlist


test = get_spglib_database(145)
print(test.coordinates)
