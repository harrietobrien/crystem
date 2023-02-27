import crystem
from spglib_database import spglib_database
from filemanager import FileManager
from crystem import CrystemList
import numpy as np
import random
import json


class Structure:
    """ store structure name here """

    # chem nom, by species, hall, space group
    def __init__(self):
        self.primVectors = None
        self.unitVectors = CrystemList.vectors
        self.crystems = CrystemList().crystems
        self.sg = SpaceGroup()
        self.name, self.crystem, self.strDict = \
            self.sg.getPtypBySpecies('CuTi')
        self.system, self.bravais = self.crystem
        [self.a, self.b, self.c] = self.getAxialLengths()
        [self.alp, self.bet, self.gam] = \
            self.getInteraxialAngles()
        self.getSiteCoordinates()
        self.constructBasisVectors()

    def getHallNumber(self):
        return self.strDict['hall#']

    def getAxialLengths(self):
        lengths = [a, b, c] = self.strDict['lengths']
        print('a: {a}, b: {b}, c: {c}'
              .format(a=a, b=b, c=c))
        return lengths

    def getInteraxialAngles(self):
        angles = [alp, bet, gam] = self.strDict['angles']
        print('alpha: {a}, beta: {b}, gamma: {g}'
              .format(a=alp, b=bet, g=gam))
        return angles

    @staticmethod
    def getWyckoffConstraints(coordinates):
        # constraints on Wyckoff positions
        D = {"x": [1., 0., 0.],
             "y": [0., 1., 0.],
             "z": [0., 0., 1.],
             "-x": [-1., 0., 0.],
             "-y": [0., -1., 0.],
             "-z": [0., 0., -1.]}
        tmp = list()  # site constraints
        assert (type(coordinates) == list)
        for vector in coordinates:
            matrix = []
            currVect = vector.replace('(', '').replace(')', '')
            for coord in currVect.split(','):
                if all(char.isdigit() or char == '/' for char in coord):
                    matrix.append([0., 0., 0.])
                else:
                    for char in coord:
                        if not (char.isalpha() or char == '-'):
                            coord = coord.replace(char, '')
                    assert(coord in D)
                    matrix.append(coord[D])
            vstack = np.column_stack(matrix)
            tmp.append(vstack)
        return tmp

    @staticmethod
    def getWyckoffTranslations(coordinates):
        # translation of the Wyckoff positions
        tmp = list()
        assert (type(coordinates) == list)
        for vector in coordinates:
            assert (type(vector) == str)
            trans = np.array([])
            vector = vector.replace('(', '').replace(')', '')
            for num in vector.split(','):
                if all((char.isalpha() or char == '-') for char in num):
                    trans = np.append(trans, 0)
                elif any(char == '+' for char in num):
                    for char in num:
                        if char.isalpha() or char == '+' or char == '-':
                            num = num.replace(char, '')
                    trans = np.append(trans, eval(num))
                elif all(char.isdigit() or char == '/' for char in num):
                    trans = np.append(trans, eval(num))
            tmp.append(trans)
        return tmp

    # returns nested dictionary of sites per/species
    def getSiteCoordinates(self):
        hall = self.getHallNumber()
        siteDict = self.sg.sortStructPositions(self.strDict)
        print('siteDict', siteDict)
        siteCoords = self.sg.getSiteCoordinates(hall, siteDict)
        print('siteCoords', siteCoords)
        generators = dict()
        for species in siteCoords:
            sites: dict = siteCoords[species]
            speciesDict = dict()
            for site in sites:
                currCoords = sites[site]['coords']
                cstr = self.getWyckoffConstraints(currCoords)
                tran = self.getWyckoffTranslations(currCoords)
                # cstr, tran: list with corresponding indices
                speciesDict[site] = (cstr, tran)
            generators[species] = speciesDict
        print(generators)
        return generators

    # lattice coordinates
    def getPrimitiveVectors(self):
        # use system to determine which crystem class --> Tetragonal
        assert (self.system in self.crystems)
        # instance of Crystem object
        sysInst = self.crystems[self.system]
        baseVects = sysInst.primitive(self.a, self.b, self.c)
        print(baseVects)
        # self.primVectors = np.vsplit(baseVects, 3)  # = [a1, a2, a3]
        # self.primVectors = [a1, a2, a3]
        return baseVects

    def constructBasisVectors(self):
        """
        primitive vectors::
        [[4.158 0.    0.   ]
         [0.    4.158 0.   ]
         [0.    0.    3.594]]
        :return:
        """
        from numpy.linalg import multi_dot
        primVectors = self.getPrimitiveVectors()
        # [a1, a2, a3] = primVectors
        # get generators + then do math
        generators = self.getSiteCoordinates()
        tmp = dict()
        for species in generators:
            tmpSpecies = dict()
            currSpecies = generators[species]
            for site in currSpecies:
                cstr = currSpecies[site][0]
                print(cstr)
                tran = currSpecies[site][1]
                # print('tran', np.dtype(tran))
                c = np.dot(primVectors, cstr) + tran
                print('c', c)
                # t = multi_dot([primVectors, tran])
                #tmpSpecies[site] = currSite
            #tmp[species] = tmpSpecies
        #print(tmp)

    # get crystal system with space or hall num
    def findCrystem(self, struct):
        # self.strDict
        return


class SpaceGroup:

    def __init__(self, hallNumber=None):
        self.fm = FileManager("data/A1.4.2.7.txt")
        # print(json.dumps(self.getHallDict(), indent=4))
        self.hallDict = self.getHallDict()
        self.spglib = spglib_database
        self.allSettings = self.getAllSettings()
        self.ptyps = self.getPrototypes()
        self.getSpaceGroupSettings(227)
        self.getAllSiteSymmetry()
        self.getPtypBySpaceGroup()
        self.getPtypsByCrystem('tP')

    def getSpaceGroupsForCrystem(self):
        return

    def getAllSiteSymmetry(self):
        siteSyms = set()
        for h in self.hallDict:
            ssList = self.spglib[str(h)]['Site Symmetry']
            for ss in ssList:
                siteSyms.add(ss)
        for i in enumerate(siteSyms):
            print(i)
        return siteSyms

    def getPrototypes(self):
        return self.fm.ODStoCSV()

    def getPtypBySpaceGroup(self):
        # print(json.dumps(self.ptyps, indent=4))
        pass

    crystem = ['a', 'm', 'o', 't', 'hh', 'th', 'h', 'c']
    bravais = ['P', 'C', 'I', 'R', 'F']
    combined = ['aP', 'mP', 'mC', 'oP', 'oC', 'oI', 'oF',
                'tP', 'tI', 'hhP', 'thP', 'hR', 'cI', 'cF']

    def getPtypsByCrystem(self, system):
        assert (system in self.ptyps)
        # print(json.dumps(self.ptyps[crystem], indent=4))
        return self.ptyps[system]

    def getPtypBySpecies(self, elements):
        for system in self.ptyps:
            currCrystem = self.ptyps[system]
            for struct in currCrystem['structs']:
                if elements in struct:
                    print(struct, currCrystem['structs'][struct])
                    structDict = currCrystem['structs'][struct]
                    return struct, system, structDict

    @staticmethod
    def sortStructPositions(struct):
        print(struct)
        siteCoords = dict()
        siteList = struct['sites']
        for speciesList in siteList:
            species, sites = speciesList.split(':')
            currSpecies = list()
            siteCoords[species] = None
            if ',' in sites:
                sites = [site.replace('(', ':').replace(')', '')
                         for site in sites.split(',')]
            else:
                sites = [sites.replace('(', ':').replace(')', '')]
            for site in sites:
                atomNo, wyckoffPos = site.split(':')
                currSpecies.append((atomNo, wyckoffPos))
            siteCoords[species] = currSpecies
        # print(structName, siteCoords)
        # hallNo = structDict['hall#']
        return siteCoords

    def getHallDict(self):
        return self.fm.readToDict()

    def getSiteCoordinates(self, hallNo, siteDict):
        hallSites = self.getHallSites(hallNo)
        coords = dict()
        for element in siteDict:
            sites = siteDict[element]
            assert (type(sites) == list)
            currSpecies = dict()
            for site in sites:
                assert (type(site) == tuple)
                noAtoms = site[0]
                # multiplicity + Wyckoff letter
                wyckoffPosition = site[1]
                currPosition = dict()
                currPosition['noAtoms'] = noAtoms
                currWyckPost = hallSites[wyckoffPosition]
                currPosition['coords'] = currWyckPost['coords']
                currPosition['symmetry'] = currWyckPost['symmetry']
                currSpecies[wyckoffPosition] = currPosition
            coords[element] = currSpecies
        print('getSiteCoords', coords)
        return coords

    def getHallSites(self, hallNumber):
        hallEntry = self.spglib[str(hallNumber)]
        multiplicity = hallEntry['Multiplicity']
        wyckoffLetter = hallEntry['Wyckoff Letter']
        coordinates = hallEntry['Coordinates']
        siteSymmetry = hallEntry['Site Symmetry']
        sites = dict()
        for i in range(len(multiplicity)):
            currMult = multiplicity[i]
            currLetter = wyckoffLetter[i]
            currSite = currMult + currLetter
            currCoords = coordinates[i]
            sites[currSite] = dict()
            sites[currSite]['coords'] = currCoords
            sites[currSite]['symmetry'] = siteSymmetry[i]
        return sites

    def getAllSettings(self):
        sgSettings = dict()
        for n in self.hallDict:
            # concise space group--> #:setting
            csg = self.hallDict[n]['N-C'].split(":")
            if len(csg) == 2:
                sgn, setting = int(csg[0]), csg[1]
            else:
                sgn, setting = int(csg[0]), None
            hallNum = dict()
            hallNum['setting'] = setting
            hallNum['sites'] = self.getHallSites(n)
            if sgn in sgSettings:
                sgSettings[sgn][n] = hallNum
            else:
                sgSettings[sgn] = dict()
                sgSettings[sgn][n] = hallNum
        # print(json.dumps(sgSettings, indent=4))
        return sgSettings

    def getSpaceGroupSettings(self, sgNumber):
        # print('Settings by Hall # for SG {sg}'.format(sg=sgNumber))
        settings = self.allSettings[sgNumber]
        print(json.dumps(settings, indent=4))
        return settings

    # centering operations for Bravais lattice type
    def getBravaisLattice(self):
        pass


sg = SpaceGroup()
st = Structure()
