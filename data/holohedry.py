"""
[abbr:crystal system]:
    [point group name]:
        [hermann-mauguin]:
            --str: full=short
            --tuple: (full, short)
        [space group range]:
            --S/E

"""
pgByCrystem = {
    "a:triclinic": {
        1: {
            "pg": "pedial",
            "hm": "1",
            "sgr": range(0, 2),
            "matrix": None
        },
        2: {
            "pg": "pinacoidal",
            "hm": "-1",
            "sgr": range(1, 3)
        }
    },
    "m:monoclinic": {
        3: {
            "pg": "sphenoidal",
            "hm": "2",
            "sgr": range(3, 6)
        },
        4: {
            "pg": "domatic",
            "hm": "m",
            "sgr": range(6, 10)
        },
        5: {
            "pg": "prismatic",
            "hm": "2/m",
            "sgr": range(10, 16)
        },
    },
    "o:orthorhombic": {
        6: {
            "pg": "rhombic disphenoidal",
            "hm": "222",
            "sgr": range(16, 25)
        },
        7: {
            "pg": "rhombic pyramidal",
            "hm": "mm2",
            "sgr": range(25, 47)
        },
        8: {
            "pg": "rhombic dipyramidal",
            "hm": ("2/m2/m2/m", "mmm"),
            "sgr": range(47, 75)
        },
    },
    "t:tetragonal": {
        9: {
            "pg": "tetragonal pyramidal",
            "hm": "4",
            "sgr": range(75, 81)
        },
        10: {
            "pg": "tetragonal disphenoidal",
            "hm": "-4",
            "sgr": range(81, 83)
        },
        11: {
            "pg": "tetragonal dipyramidal",
            "hm": "4/m",
            "sgr": range(83, 89)
        },
        12: {
            "pg": "tetragonal trapezohedral",
            "hm": "422",
            "sgr": range(89, 99)
        },
        13: {
            "pg": "ditetragonal pyramidal",
            "hm": "4mm",
            "sgr": range(99, 111)
        },
        14: {
            "pg": "tetragonal scalenohedral",
            "hm": "-42m",
            "sgr": range(111, 123)
        },
        15: {
            "pg": "ditetragonal dipyramidal",
            "hm": ("4/m2/m2/m", "4/mmm"),
            "sgr": range(123, 143)
        }
    },
    "h:trigonal": {
        16: {
            "pg": "trigonal pyramidal",
            "hm": "3",
            "sgr": range(143, 147)
        },
        17: {
            "pg": "rhombohedral",
            "hm": "-3",
            "sgr": range(147, 149)
        },
        18: {
            "pg": "trigonal trapezohedral",
            "hm": "32",
            "sgr": range(149, 156)
        },
        19: {
            "pg": "ditrigonal pyramidal",
            "hm": "3m",
            "sgr": range(156, 162)
        },
        20: {
            "pg": "ditrigonal scalenohedral",
            "hm": ("-32/m", "-3m"),
            "sgr": range(162, 168)
        },
    },
    "h:hexagonal": {
        21: {
            "pg": "hexagonal pyramidal",
            "hm": "6",
            "sgr": range(168, 174)
        },
        22: {
            "pg": "trigonal dipyramidal",
            "hm": "-6",
            "sgr": range(174, 175)
        },
        23: {
            "pg": "hexagonal dipyramidal",
            "hm": "6/m",
            "sgr": range(175, 177)
        },
        24: {
            "pg": "hexagonal trapezohedral",
            "hm": "622",
            "sgr": range(177, 183)
        },
        25: {
            "pg": "dihexagonal pyramidal",
            "hm": "6mm",
            "sgr": range(183, 187)
        },
        26: {
            "pg": "ditrigonal dipyramidal",
            "hm": "-6m2",
            "sgr": range(187, 191)
        },
        27: {
            "pg": "dihexagonal dipyramidal",
            "hm": ("6/m2/m2/m", "6/mmm"),
            "sgr": range(191, 195)
        },
    },
    "c:cubic": {
        28: {
            "pg": "tetartoidal",
            "hm": "23",
            "sgr": range(195, 200)
        },
        29: {
            "pg": "diploidal",
            "hm": ("2/m-3", "m-3"),
            "sgr": range(200, 207)
        },
        30: {
            "pg": "gyroidal",
            "hm": "432",
            "sgr": range(207, 215)
        },
        31: {
            "pg": "hextetrahedral",
            "hm": "-43m",
            "sgr": range(215, 221)
        },
        32: {
            "pg": "hexoctahedral",
            "hm": ("4/m-32/m", "m-3m"),
            "sgr": range(221, 231)
        },
    }
}


class Holohedry:
    def __init__(self):
        pass
