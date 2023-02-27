import numpy as np
import sys

sys.path.append('data/holohedry.py')


class CrystemList:

    def __init__(self):
        self.crystems = self.getCrystems()

    @staticmethod
    def getCrystems() -> list:
        """
        :rtype: dict()
        """
        crystems = dict()

        class CrystemVectors:
            # orthogonal vectors with unit length
            x = np.array([1, 0, 0])  # xˆ
            y = np.array([0, 1, 0])  # yˆ
            z = np.array([0, 0, 1])  # zˆ

        v = CrystemVectors()
        CrystemList.vectors = v

        class Triclinic(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSym = 'a'

            # in terms of 3D cell parameters
            @staticmethod
            def projectionDirection():
                # requires --> a, b, alp, bet, gam
                # noinspection PyUnusedLocal
                def _001(a, b, alp, bet, gam):
                    a_prime = a * np.sin(bet)
                    b_prime = b * np.sin(alp)
                    cos_gam_star = ((np.cos(alp) * np.cos(bet) - np.cos(gam))
                                    (np.sin(alp) * np.sin(bet)))
                    gam_star = np.arccos(cos_gam_star)
                    gam_prime = 180 - gam_star
                    return a_prime, b_prime, gam_prime

                projectionDirection._001 = _001

                # requires --> b, c, alp, bet, gam
                # noinspection PyUnusedLocal
                def _100(b, c, alp, bet, gam):
                    a_prime = b * np.sin(gam)
                    b_prime = c * np.sin(bet)
                    cos_alp_star = ((np.cos(bet) * np.cos(gam) - np.cos(alp))
                                    (np.sin(bet) * np.sin(gam)))
                    alp_star = np.arccos(cos_alp_star)
                    gam_prime = 180 - alp_star
                    return a_prime, b_prime, gam_prime

                projectionDirection._100 = _100

                # requires --> a, c, alp, bet, gam
                # noinspection PyUnusedLocal
                def _010(a, c, alp, bet, gam):
                    a_prime = c * np.sin(alp)
                    b_prime = a * np.sin(gam)
                    cos_bet_star = (np.cos(gam) * np.cos(alp) - np.cos(bet)) \
                        (np.sin(gam) * np.sin(alp))
                    bet_star = np.arccos(cos_bet_star)
                    gam_prime = 180 - bet_star
                    return a_prime, b_prime, gam_prime

                projectionDirection._010 = _010

            def primitive(self, a, b, c, alp, bet, gam) -> np.vstack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                param float alp: interaxial angle
                param float bet: interaxial angle
                param float gam: interaxial angle
                """
                import math
                cx = c * np.cos(bet)
                print('c', c)
                print('cx np', cx)
                print('cx math', c * math.cos(bet))
                cy = (c * (np.cos(alp) - np.cos(bet) * np.cos(gam))) / np.sin(gam)
                print('cy', cy)
                hello = math.pow(c, 2) - math.pow(cx, 2) - math.pow(cy, 2)
                print(' math.pow(c, 2)', math.pow(c, 2))
                print('math.pow(cx, 2)', math.pow(cx, 2))
                print('math.pow(cy, 2)', math.pow(cy, 2))
                print('hello', hello)
                cz = math.sqrt(math.pow(c, 2) - math.pow(cx, 2) - math.pow(cy, 2))
                a1 = a * self.x
                a2 = b * np.cos(gam) * self.x + b * np.sin(gam) * self.y
                a3 = cx * self.x + cy * self.y + cz * self.z

                def getUnitCellVolume():
                    return _a * _b * cz * np.sin(gam)

                # np.vstack([a1, a2, a3]).T works as well
                return np.column_stack([a1, a2, a3])

        ano = Triclinic()
        aTmp = dict()
        aTmp['name'] = 'Triclinic (anorthic)'
        aTmp['settings'] = {'P': ['primitive']}
        reqParams = ['a', 'b', 'c', 'alp', 'bet', 'gam']
        aTmp['settings']['P'].append(reqParams)
        aTmp['projections'] = ['_001', '_100', '_010']
        aTmp['instance'] = ano
        crystems[ano.sysSym] = aTmp

        class Monoclinic(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSym = 'm'
                self.unique_axis = 'b'

            # in terms of 3D cell parameters
            def projectionDirection(self):

                # requires --> a, b, angle (bet or gam)
                # noinspection PyUnusedLocal
                def _001(a, b, angle):
                    if self.unique_axis == 'b':
                        # angle is beta
                        a_prime = a * np.sin(angle)
                        b_prime = b
                        gam_prime = 90
                        return a_prime, b_prime, gam_prime
                    else:
                        assert (self.unique_axis == 'c')
                        # angle is gamma
                        a_prime = a
                        b_prime = b
                        gam_prime = angle
                        return a_prime, b_prime, gam_prime

                projectionDirection._001 = _001

                # requires --> b, c, angle (bet or gam)
                # noinspection PyUnusedLocal
                def _100(b, c, angle):
                    if self.unique_axis == 'b':
                        # angle is beta
                        a_prime = b
                        b_prime = c * np.sin(angle)
                        gam_prime = 90
                        return a_prime, b_prime, gam_prime
                    else:
                        assert (self.unique_axis == 'c')
                        # angle is gamma
                        a_prime = b * np.sin(angle)
                        b_prime = c
                        gam_prime = 90
                        return a_prime, b_prime, gam_prime

                projectionDirection._100 = _100

                # requires --> a, c, angle (bet or gam)
                # noinspection PyUnusedLocal
                def _010(a, c, angle):
                    if self.unique_axis == 'b':
                        # angle is beta
                        a_prime = c
                        b_prime = a
                        gam_prime = angle
                        return a_prime, b_prime, gam_prime
                    else:
                        assert (self.unique_axis == 'c')
                        # angle is gamma
                        a_prime = c
                        b_prime = a * np.sin(angle)
                        gam_prime = 90
                        return a_prime, b_prime, gam_prime

                projectionDirection._010 = _010

            def primitive(self, a, b, c, bet) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                param float bet: interaxial angle
                """

                a1 = a * self.x
                a2 = b * self.y
                a3 = c * np.cos(bet) * self.x + c * np.sin(bet) * self.z

                def getUnitCellVolume():
                    return a * b * c * np.sin(bet)

                return np.column_stack([a1, a2, a3])

            def baseCentered(self, a, b, c, bet) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                param float bet: interaxial angle
                """
                a1 = 0.5 * a * self.x - 0.5 * b * self.y
                a2 = 0.5 * a * self.x + 0.5 * b * self.y
                a3 = c * np.cos(bet) * self.x + c * np.sin(bet) * self.z

                def getUnitCellVolume():
                    return 0.5 * a * b * c * np.sin(bet)

                return np.column_stack([a1, a2, a3])

        mon = Monoclinic()
        mTmp = dict()
        mTmp['name'] = 'Monoclinic'
        mTmp['settings'] = {'P': ['primitive'],
                            'C': ['baseCentered']}
        reqParams = ['a', 'b', 'c', 'bet']
        mTmp['settings']['P'].append(reqParams)
        mTmp['settings']['C'].append(reqParams)
        mTmp['projections'] = ['_001', '_100', '_010']
        mTmp['instance'] = mon
        crystems[mon.sysSym] = mTmp

        class Orthorhombic(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSym = 'o'

            # in terms of 3D cell parameters
            @staticmethod
            def projectionDirection():
                # requires --> a, b
                # noinspection PyUnusedLocal
                def _001(a, b):
                    a_prime = a
                    b_prime = b
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirection._001 = _001

                # requires --> b, c
                # noinspection PyUnusedLocal
                def _100(b, c):
                    a_prime = b
                    b_prime = c
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirection._100 = _100

                # requires --> c, a
                # noinspection PyUnusedLocal
                def _010(a, c):
                    a_prime = c
                    b_prime = a
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirection._010 = _010

            def primitive(self, a, b, c) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                """
                a1 = a * self.x
                a2 = b * self.y
                a3 = c * self.z

                def getUnitCellVolume():
                    return a * b * c

                return np.column_stack([a1, a2, a3])

            def baseCentered(self, a, b, c) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                """
                # SGs with 'C' (translation in the a-b plane)
                a1 = 0.5 * a * self.x - 0.5 * b * self.y
                a2 = 0.5 * a * self.x + 0.5 * b * self.y
                a3 = c * self.z
                # SGs with 'A' (translation in the b-c plane)

                def getUnitCellVolume():
                    return a * b * c / 2.

                return np.column_stack([a1, a2, a3])

            def bodyCentered(self, a, b, c) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                """
                a1 = -0.5 * a * self.x + 0.5 * b * self.y + 0.5 * c * self.z
                a2 = 0.5 * a * self.x - 0.5 * b * self.y + 0.5 * c * self.z
                a3 = 0.5 * a * self.x + 0.5 * b * self.y - 0.5 * c * self.z

                def getUnitCellVolume():
                    return a * b * c / 2.

                return np.column_stack([a1, a2, a3])

            def faceCentered(self, a, b, c) -> np.column_stack:
                """
                param float a: axial length
                param float b: axial length
                param float c: axial length
                """
                a1 = 0.5 * b * self.y + 0.5 * c * self.z
                a2 = 0.5 * a * self.x + 0.5 * c * self.z
                a3 = 0.5 * a * self.x + 0.5 * b * self.y

                def getUnitCellVolume():
                    return a * b * c / 4.

                return np.column_stack([a1, a2, a3])

        ort = Orthorhombic()
        oTmp = dict()
        oTmp['name'] = 'Orthorhombic'
        oTmp['settings'] = {'P': ['primitive'],
                            'C': ['baseCentered'],
                            'I': ['bodyCentered'],
                            'F': ['faceCentered']}
        reqParams = ['a', 'b', 'c']
        for setting in oTmp['settings']:
            oTmp['settings'][setting].append(reqParams)
        oTmp['projections'] = ['_001', '_100', '_010']
        oTmp['instance'] = ort
        crystems[ort.sysSym] = oTmp

        class Tetragonal(CrystemVectors):
            def __init__(self):
                super().__init__()
                self.sysSym = 't'

            @staticmethod
            def projectionDirections():
                # requires --> a
                # noinspection PyUnusedLocal
                def _001(a):
                    a_prime = a
                    b_prime = a
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._001 = _001

                # requires --> a, c
                # noinspection PyUnusedLocal
                def _100(a, c):
                    a_prime = a
                    b_prime = c
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._100 = _100

                # requires --> a, c
                # noinspection PyUnusedLocal
                def _110(a, c):
                    a_prime = (a / 2) * np.sqrt(2)
                    b_prime = c
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._110 = _110

            def primitive(self, a, c) -> np.column_stack:
                """
                param float a: axial length
                param float c: axial length
                """
                a1 = a * self.x
                a2 = a * self.y
                a3 = c * self.z

                def getUnitCellVolume():
                    return a ** 2 * c

                return np.column_stack([a1, a2, a3])

            def bodyCentered(self, a, c) -> np.column_stack:
                """
                param float a: axial length
                param float c: axial length
                """
                a1 = -0.5 * a * self.x + 0.5 * a * self.y + 0.5 * c * self.z
                a2 = 0.5 * a * self.x - 0.5 * a * self.y + 0.5 * c * self.z
                a3 = 0.5 * a * self.x + 0.5 * a * self.y - 0.5 * c * self.z

                def getUnitCellVolume():
                    return (a ** 2 * c) / 2.

                return np.vstack([a1, a2, a3])

        tet = Tetragonal()
        tTmp = dict()
        tTmp['name'] = 'Tetragonal'
        tTmp['settings'] = {'P': ['primitive'],
                            'I': ['bodyCentered']}
        tTmp['settings']['P'].append(['a', 'c'])
        tTmp['settings']['I'].append(['a', 'c'])
        tTmp['projections'] = ['_001', '_100', '_110']
        tTmp['instance'] = tet
        crystems[tet.sysSym] = tTmp

        class Hexagonal(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSym = 'h'

            @staticmethod
            def projectionDirections():
                # requires --> a
                # noinspection PyUnusedLocal
                def _001(a):
                    a_prime = a
                    b_prime = a
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._001 = _001

                # requires --> a, c
                # noinspection PyUnusedLocal
                def _100(a, c):
                    a_prime = (a / 2) * np.sqrt(3)
                    b_prime = c
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._100 = _100

                # requires --> a, c
                # noinspection PyUnusedLocal
                def _210(a, c):
                    a_prime = a / 2
                    b_prime = c
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._210 = _210

            def primitive(self, a, c) -> np.column_stack:
                """
                param float a: axial length
                param float c: axial length
                """
                a1 = 0.5 * a * self.x - (np.sqrt(3) / 2.0) * a * self.y
                a2 = 0.5 * a * self.x + (np.sqrt(3) / 2.0) * a * self.y
                a3 = c * self.z

                def getUnitCellVolume():
                    return (np.sqrt(3) / 2) * a ** 2 * c

                return np.column_stack([a1, a2, a3])

        hex = Hexagonal()
        hTmp = dict()
        hTmp['name'] = 'Hexagonal'
        hTmp['settings'] = {'P': ['primitive']}
        hTmp['settings']['P'].append(['a', 'c'])
        hTmp['projections'] = ['_001', '_100', '_210']
        hTmp['instance'] = hex
        crystems[hex.sysSym] = hTmp

        class Trigonal(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSymbol = 'r'

            @staticmethod
            def projectionDirections():
                # requires --> a, alp
                # noinspection PyUnusedLocal
                def _111(a, alp):
                    a_prime = (2 / np.sqrt(3)) * a * np.sin(alp / 2)
                    b_prime = (2 / np.sqrt(3)) * a * np.sin(alp / 2)
                    gam_prime = 120
                    return a_prime, b_prime, gam_prime

                projectionDirections._111 = _111

                # requires --> a, alp
                # noinspection PyUnusedLocal
                def _1n10(a, alp):
                    a_prime = a * np.cos(alp / 2)
                    b_prime = a
                    cos_rho = np.cos(alp) / np.cos(alp / 2)
                    gam_prime = np.arccos(cos_rho)
                    return a_prime, b_prime, gam_prime

                projectionDirections._1n10 = _1n10

                # requires --> a, alp
                # noinspection PyUnusedLocal
                def _n211(a, alp):
                    a_prime = (1 / np.sqrt(3)) * a * np.sqrt(1 + 2 * np.cos(alp))
                    b_prime = a * np.sin(alp / 2)
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._n211 = _n211

            # hP --> Hexagonal Bravais Lattice
            def primitive(self, a, alp) -> np.column_stack:
                """
                param float a: axial length
                param float alp: interaxial angle
                """
                assert (alp != np.pi / 2.0)
                a1 = 0.5 * a * self.x - (np.sqrt(3) / 2) * a * self.y
                a2 = 0.5 * a * self.x + (np.sqrt(3) / 2) * a * self.y
                a3 = a * self.z

                def getUnitCellVolume():
                    return (np.sqrt(3) / 2) * a ** 2 * c

                return np.column_stack([a1, a2, a3])

            # hR --> Rhombohedral Bravais Lattice
            def rhombohedral(self, a, c) -> np.column_stack:
                """
                param float a: axial length
                param float c: axial length
                """
                a1 = 0.5 * a * self.x - 1 / (2 * np.sqrt(3)) * a * self.y + (1 / 3) * c * self.z
                a2 = 1 / np.sqrt(3) * a * self.y + (1 / 3) * c * self.z
                a3 = -0.5 * a * self.x - 1 / (2 * np.sqrt(3)) * a * self.y + (1 / 3) * c * self.z

                def getUnitCellVolume():
                    return (2 / np.sqrt(3)) * a ** 2 * c

                return np.column_stack([a1, a2, a3])

        tri = Trigonal()
        rTmp = dict()
        rTmp['name'] = 'Trigonal'
        rTmp['settings'] = {'P': ['primitive'],
                            'R': ['rhombohedral']}
        rTmp['settings']['P'].append(['a', 'alp'])
        rTmp['settings']['R'].append(['a', 'c'])
        rTmp['projections'] = ['_111', '_1n10', '_n211']
        rTmp['instance'] = tri
        crystems[tri.sysSymbol] = rTmp

        class Cubic(CrystemVectors):

            def __init__(self):
                super().__init__()
                self.sysSym = 'c'

            @staticmethod
            def projectionDirections():
                # requires --> a
                # noinspection PyUnusedLocal
                def _001(a):
                    a_prime = a
                    b_prime = a
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._001 = _001

                # requires --> a
                # noinspection PyUnusedLocal
                def _111(a):
                    a_prime = a * np.sqrt(2 / 3)
                    b_prime = a * np.sqrt(2 / 3)
                    gam_prime = 120
                    return a_prime, b_prime, gam_prime

                projectionDirections._111 = _111

                # requires --> a
                # noinspection PyUnusedLocal
                def _110(a):
                    a_prime = (a / 2) * np.sqrt(2)
                    b_prime = a
                    gam_prime = 90
                    return a_prime, b_prime, gam_prime

                projectionDirections._110 = _110

            def primitive(self, a) -> np.vstack:
                """
                :param float a: axial length
                """
                a1 = a * self.x
                a2 = a * self.y
                a3 = a * self.z

                def getUnitCellVolume():
                    return a ** 3

                return np.column_stack([a1, a2, a3])

            def bodyCentered(self, a) -> np.column_stack:
                """
                param float a: axial length
                """
                a1 = -0.5 * a * self.x + 0.5 * a * self.y + 0.5 * a * self.z
                a2 = 0.5 * a * self.x - 0.5 * a * self.y + 0.5 * a * self.z
                a3 = 0.5 * a * self.x + 0.5 * a * self.y - 0.5 * a * self.z
                # a1 = a * self.x
                # a2 = a * self.y
                # a3 = (a / 2) * (self.x + self.y + self.z)

                def getUnitCellVolume():
                    return a ** 3 / 2

                return np.column_stack([a1, a2, a3])

            def faceCentered(self, a) -> np.column_stack:
                """
                param float a: axial length
                """
                a1 = 0.5 * a * self.y + 0.5 * a * self.z
                a2 = 0.5 * a * self.x + 0.5 * a * self.z
                a3 = 0.5 * a * self.x + 0.5 * a * self.y

                def getUnitCellVolume():
                    return a ** 3 / 4

                return np.column_stack([a1, a2, a3])

        iso = Cubic()
        cTmp = dict()
        cTmp['name'] = 'Cubic (Isometric)'
        cTmp['settings'] = {'P': ['primitive'],
                            'I': ['bodyCentered'],
                            'F': ['faceCentered']}
        for setting in cTmp['settings']:
            cTmp['settings'][setting].append(['a'])
        cTmp['projections'] = ['_001', '_111', '_110']
        cTmp['instance'] = iso
        crystems[iso.sysSym] = cTmp

        return crystems


class Test:

    def __init__(self):
        self.systems = CrystemList().crystems
        self.a, self.b, self.c = None, None, None
        self.alp, self.bet, self.gam = None, None, None

    def setConstants(self, system):
        constants = {
            # aP --> Cf
            'a': [3.307, 7.412, 2.793, 89.06, 85.15, 85.7],
            # mP --> NiTi
            'm': [2.884, 4.106, 4.667, 90, 82.062, 90],
            # oC --> HCl
            'o': [5.825, 5.505, 5.373, 90, 90, 90],
            # tI --> MoSi2
            't': [3.206, 3.206, 7.848, 90, 90, 90],
            # hP --> ZnS
            'h': [3.823, 3.823, 6.261, 90, 90, 120],
            # hR --> CuPt
            'r': [3.130, 3.130, 14.980, 90, 90, 120],
            # cF --> BiF3
            'c': [5.853, 5.853, 5.853, 90, 90, 90],
        }
        attrs = ['a', 'b', 'c', 'alp', 'bet', 'gam']
        load = constants[system]
        for i in range(len(attrs)):
            attr = attrs[i]
            setattr(self, attr, load[i])

    @staticmethod
    def wrapper(method, args):
        return method(*args)

    def testAll(self):
        for symbol in self.systems:
            self.testSystem(symbol)

    def testSystem(self, system):
        self.setConstants(system)
        systemDict = self.systems[system]
        settings = systemDict['settings']
        inst = systemDict['instance']
        for bravais in settings:
            print(systemDict['name'], bravais)
            func, reqParams = settings[bravais]
            params = list()
            for i in reqParams:
                params.append(getattr(self, i))
            method = getattr(inst, func)
            basis = self.wrapper(method, params)
            print(basis)


test = Test()
test.testSystem('a')
# print(test)
