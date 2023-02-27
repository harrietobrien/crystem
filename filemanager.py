import json


class FileManager:

    def __init__(self, file):
        self.fname = file
        self.file = None
        self.descr = None
        self.hallNums = dict()

    def parseThis(self):
        # retrieve explicit generators from tu5004sup1.txt
        with open('tu5004sup1.txt', 'r') as file:
            self.file = file.read().splitlines()
            for line in self.file:
                print(line)

    def saveToFile(self):
        with open('hall_out.txt', 'w') as file:
            for k in self.hallNums:
                v = self.hallNums[k]
                # concise space group
                csg = '\tconcise SG\t' + v['N-C']
                # H-M entry
                hme = '\tH-M entry\t' + v['H-M']
                out = [str(k), csg, '\n', hme, '\n']
                for i in out:
                    if i.isdigit():
                        file.write(str(i))
                        file.write('\n')
                    else:
                        file.write(i)

    def printDict(self):
        self.getDescr()
        print(json.dumps(self.hallNums, indent=4))

    def printODS(self):
        print(json.dumps(self.ODStoCSV(), indent=4))

    def readToDict(self):
        with open(self.fname, 'r') as reader:
            self.file = reader.read().splitlines()
            lines = [self.file[i] for i in range(len(self.file))
                     if len(self.file[i].split()) == 3]
            for i in range(len(lines)):
                assert (len(lines) == 530)
                entry = lines[i].split()
                spgDict = dict()
                # N-C --> (space group) Number-Code
                spgDict["N-C"] = entry[0]
                # H-M --> Hermann–Mauguin notation
                spgDict["H-M"] = entry[1]
                # H-S --> Hall (symbol) notation
                spgDict["H-S"] = entry[2]
                self.hallNums[i + 1] = spgDict
        return self.hallNums

    def getDescr(self):
        if self.file is not None:
            for i in range(len(self.file)):
                line = self.file[i]
                if line.startswith("#"):
                    print(line.replace("#", ''))

    '''constants for testing'''

    @staticmethod
    def ODStoCSV():
        from pyexcel_ods import get_data
        # data: OrderedDict
        data = get_data("constants.ods")
        # get rid of empty rows at the end
        materials = dict()
        for sheet in data:
            tmp = dict()
            currBravais = None
            for row in data[sheet][3:]:
                if not row:
                    assert (currBravais == 'cF')
                    materials[currBravais] = tmp
                    break
                elif type(row[0]) is int:
                    # aP, mP, mC, oP, oC, oI, oF, tP, tI, hhP, thP, hR, cI cF
                    byBravais, crystem = row[4].split(":")
                    if byBravais != currBravais:
                        if currBravais is not None:
                            materials[currBravais] = tmp
                            tmp = dict()
                        currBravais = byBravais
                        tmp['name'] = crystem
                        tmp['structs'] = dict()
                else:
                    assert (type(row[0]) is str)
                    name = row[0].strip()
                    structs = tmp['structs']
                    struct = dict()
                    noSpecies, noAtoms, sites = row[1], row[2], row[3]
                    species = [i.strip() for i in sites.split(";")
                               if not i or not i.isspace()]
                    assert (len(species) == int(noSpecies))
                    struct['#atoms'] = int(noAtoms)
                    struct['#species'] = int(noSpecies)
                    struct['sites'] = species
                    # axial lengths of primitive vectors
                    struct['lengths'] = [row[5], row[6], row[7]]
                    # interaxial angles between axial lengths
                    struct['angles'] = [row[8], row[9], row[10]]
                    struct['H-M'] = row[11].strip()
                    struct['CSG#'] = str(row[12]).strip()
                    struct['hall#'] = row[13]
                    structs[name] = struct
        # only json object prints the unicode
        # print(json.dumps(materials, indent=4))
        # print(materials)
        return materials

test = FileManager("data/A1.4.2.7.txt")
# test.readToDict()
# test.regularPrint()
# print("hi")
# test.printODS()
# test.parseThis()
