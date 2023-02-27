import matplotlib
import numpy as np
import json

matplotlib.use('WXAgg')
import wx
from pubsub import pub
from util import MyFileDropTarget
from filemanager import FileManager
from crystem import CrystemList


class ControlPanel(wx.Panel):
    subscripts = ['\u2081', '\u2082', '\u2083']
    latticeParams = dict((key, dict()) for key in list('abcαβγ'))
    angleDescr = {'α': 'b ^ c', 'β': 'c ^ a', 'γ': 'a ^ b'}
    crystemDict = CrystemList.getCrystems()
    prototypes = FileManager.ODStoCSV()
    print('prototypes\n', prototypes)
    symDirs = {'Primary': ['001', '111'],
               'Secondary': ['100', '111', '11\u03050'],
               'Tertiary': ['010', '110', '210', '2\u030511']}

    blackIsh = "#262322"
    creamIsh = "#F2E5D7"

    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        self.slider = None
        self.ptSzText = ''
        self.ptSize = 1
        self.edges: int = 0
        self.cpColor = ControlPanel.blackIsh
        self.textColor = ControlPanel.creamIsh
        self.SetBackgroundColour(self.cpColor)

        self.prototypes = ControlPanel.prototypes

        self.structures = dict()
        self.structCount = len(self.structures)
        self.currentCrystem = 'no current structure selection'

        self._14 = wx.Font(wx.FontInfo(14).FaceName("Helvetica").Light())
        self._12 = wx.Font(wx.FontInfo(12).FaceName("Helvetica").Light())
        self._10 = wx.Font(wx.FontInfo(10).FaceName("Helvetica").Light())
        self._bold = wx.Font(wx.FontInfo(14).FaceName("Helvetica").Bold())
        self._italic = wx.Font(wx.FontInfo(12).FaceName("Helvetica").Italic())

        self.saveAs = None
        self.crystemButtons = list()
        # lattice constants
        self.a, self.b, self.c = None, None, None
        self.alp, self.bet, self.gam = None, None, None
        # text control object storage
        self.axialLengths: list[object] = list()
        self.interaxialAngles: list[object] = list()

        systems = list(ControlPanel.crystemDict.keys())
        self.protoSystems = wx.ListBox(self, choices=systems, size=(70, -1), style=wx.LB_SINGLE)
        self.Bind(wx.EVT_LISTBOX, self.onCrystemListBox, self.protoSystems)
        self.currBravaisOpts = []
        # fill bravais box based on
        self.bravaisBox = wx.ListBox(self, choices=self.currBravaisOpts,
                                     size=(160, -1), style=wx.LB_SINGLE)
        self.Bind(wx.EVT_LISTBOX, self.onBravaisListBox, self.bravaisBox)
        self.currStructOpts = []
        self.structBox = wx.ListBox(self, choices=self.currStructOpts, size=(150, -1), style=wx.LB_SINGLE)
        self.Bind(wx.EVT_LISTBOX, self.onStructListBox, self.structBox)

        self.lpDict = dict()

        self.rotation_angle = ''
        self.u, self.v, self.w = '', '', ''
        self.axis = ''
        self.translationVector = ''

        self.textColor = wx.Colour(ControlPanel.creamIsh)
        self.font = wx.Font(wx.FontInfo(18).FaceName("Helvetica"))
        self.ctrlInterface()

    def onSlider(self, event):
        obj = event.GetEventObject()
        self.ptSize = obj.GetValue()
        pub.sendMessage('ptSizeListener',
                        message=self.ptSize,
                        listener='ptSizeListener')

    def getBoxChoices(self):
        protoByFamily = {key: dict() for key in self.crystemDict.keys()}
        # regroup by crystal family
        for bl in self.prototypes:
            currBL = self.prototypes[bl]
            sys, brav = bl[0], bl[1]
            protoByFamily[sys][brav] = currBL['structs']
        return protoByFamily

    def ctrlInterface(self):
        ctrlVbox = wx.BoxSizer(wx.VERTICAL)

        # cphBox = control panel horizontal box
        cphBox = wx.FlexGridSizer(cols=3, rows=1, vgap=0, hgap=10)
        cphBox.SetFlexibleDirection(wx.HORIZONTAL)

        crystemLabel = wx.StaticText(self, label='Select crystal system \u2192 '
                                                 'Input parameters')
        lattPamLabel = wx.StaticText(self, label='Input parameters \u2192 '
                                                 'Inspect crystal system')
        crystemLabel.SetFont(self._14)
        lattPamLabel.SetFont(self._14)

        lpBox = wx.StaticBox(self, -1, "Enter Lattice Parameters")
        lpvSizer = wx.StaticBoxSizer(lpBox, wx.VERTICAL)
        lpvSizer.Add(lattPamLabel, flag=wx.ALL, border=5)
        lphSizer = wx.BoxSizer(wx.HORIZONTAL)
        params = list(ControlPanel.latticeParams.keys())
        lengths = params[0:len(params) // 2]
        angles = params[len(params) // 2:]
        # INPUT AXIAL LENGTHS
        alvBox = wx.BoxSizer(wx.VERTICAL)
        alvBox.Add(wx.StaticText(self, label="Axial Lengths"))
        for length in lengths:
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            text = wx.StaticText(self, label=length)
            hbox.Add(text, proportion=1, flag=wx.ALL, border=5)
            nameTag = "lp:al:{len}".format(len=length)
            inputCtrl = wx.TextCtrl(self, value='', name=nameTag)
            unit = wx.StaticText(self, label='Å')
            hbox.Add(inputCtrl, proportion=4, flag=wx.ALL, border=5)
            self.Bind(wx.EVT_TEXT_ENTER, self.parameterSave, id=wx.ID_ANY)
            self.axialLengths.append(inputCtrl)
            hbox.Add(unit, flag=wx.ALL, border=5)
            alvBox.Add(hbox)
        lphSizer.Add(alvBox, proportion=1, flag=wx.ALL, border=5)
        # INPUT INTERAXIAL ANGLES
        iavBox = wx.BoxSizer(wx.VERTICAL)
        iavBox.Add(wx.StaticText(self, label="Interaxial Angles"))
        for angle in angles:
            hbox = wx.BoxSizer(wx.HORIZONTAL)
            descr = ControlPanel.angleDescr[angle]
            angleLabel = "º\t[ {dsc} ]".format(dsc=descr)
            text = wx.StaticText(self, label=angle)
            hbox.Add(text, proportion=1, flag=wx.ALL, border=5)
            inputCtrl = wx.TextCtrl(self, value='', name="lp:ia")
            hbox.Add(inputCtrl, proportion=4, flag=wx.ALL, border=5)
            unit = wx.StaticText(self, label=angleLabel)
            hbox.Add(unit, flag=wx.ALL, border=5)
            self.Bind(wx.EVT_TEXT_ENTER, self.parameterSave, id=wx.ID_ANY)  # ?
            self.interaxialAngles.append(inputCtrl)
            iavBox.Add(hbox)
        lphSizer.Add(iavBox, proportion=1, flag=wx.ALL, border=5)
        lpvSizer.Add(lphSizer)

        ctrlBox = wx.BoxSizer(wx.VERTICAL)
        ndBox = wx.BoxSizer(wx.HORIZONTAL)  # for nd lattice generation
        ucBox = wx.BoxSizer(wx.HORIZONTAL)  # primitive / conventional unit cell
        ucBox.Add(wx.StaticText(self, label='Unit cell:'),
                  flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        # primitive unit cell button
        pucBtn = wx.Button(self, -1, 'Primitive', style=wx.BU_EXACTFIT)
        ucBox.Add(pucBtn, flag=wx.LEFT | wx.RIGHT, border=5)
        # conventional unit cell button
        cucBtn = wx.Button(self, -1, 'Conventional', style=wx.BU_EXACTFIT)
        ucBox.Add(cucBtn, flag=wx.LEFT | wx.RIGHT, border=5)
        # primitive & conventional
        bucBtn = wx.Button(self, -1, 'Show Both', style=wx.BU_EXACTFIT)
        ucBox.Add(bucBtn, flag=wx.LEFT | wx.RIGHT, border=5)

        ndBox.Add(wx.StaticText(self, label='Edges: n'),
                  flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        ndCtrl = wx.SpinCtrl(self, value='10', size=(60, 20), style=wx.SP_VERTICAL)
        ndCtrl.SetRange(1, 20)
        ndBox.Add(ndCtrl, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        ndBox.Add(wx.StaticText(self, label='— dimensional'),
                  flag=wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        ndBtn = wx.ToggleButton(self, -1, 'Generate lattice', style=wx.BU_EXACTFIT)
        ndBox.Add(ndBtn, flag=wx.LEFT | wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        ctrlBox.Add(ucBox, flag=wx.TOP | wx.BOTTOM, border=5)
        ctrlBox.Add(ndBox, flag=wx.TOP | wx.BOTTOM, border=5)
        lpvSizer.Add(ctrlBox, flag=wx.TOP | wx.BOTTOM)
        saveBox = wx.BoxSizer(wx.HORIZONTAL)
        saveBox.Add(wx.StaticText(self, label='Save struct as'),
                    flag=wx.ALL | wx.EXPAND, border=5)
        self.saveAs = wx.TextCtrl(self, value='', size=(120, 20))
        saveBtn = wx.Button(self, -1, 'Save', style=wx.BU_EXACTFIT)
        resetBtn = wx.Button(self, -1, 'Reset', style=wx.BU_EXACTFIT)
        saveBox.Add(self.saveAs, flag=wx.ALL | wx.EXPAND, border=5)
        saveBox.Add(saveBtn, flag=wx.ALL | wx.EXPAND, border=5)
        saveBtn.Bind(wx.EVT_BUTTON, self.onSaveStructure, id=wx.ID_ANY)
        saveBox.Add(resetBtn, flag=wx.ALL | wx.EXPAND, border=5)
        resetBtn.Bind(wx.EVT_BUTTON, self.onResetTextCtrls, id=wx.ID_ANY)
        lpvSizer.Add(saveBox, flag=wx.EXPAND)

        # VIEW CRYSTAL SYSTEM
        csBox = wx.StaticBox(self, -1, "Restrict by Crystal System")
        csvSizer = wx.StaticBoxSizer(csBox, wx.VERTICAL)
        csvSizer.Add(crystemLabel, flag=wx.ALL, border=5)
        innerBox = wx.FlexGridSizer(cols=3, rows=8, vgap=0, hgap=5)
        innerBox.SetFlexibleDirection(wx.HORIZONTAL)
        row1 = wx.StaticText(self, label="")
        row1.SetFont(self._14)
        row2 = wx.StaticText(self, label="Crystal System")
        row2.SetFont(self._14)
        row3 = wx.StaticText(self, label="Bravais Lattices")
        row3.SetFont(self._14)
        innerBox.Add(row1)
        innerBox.Add(row2)
        innerBox.Add(row3)
        for system in ControlPanel.crystemDict:
            symbol = wx.StaticText(self, label=system)
            symbol.SetFont(self._italic)
            innerBox.Add(symbol, proportion=0.25, flag=wx.ALL | wx.EXPAND, border=5)
            currSystem = ControlPanel.crystemDict[system]
            # if system == 'r': system = 'h'
            settings = currSystem['settings']
            label = wx.StaticText(self, label=currSystem['name'] + '\t')
            innerBox.Add(label, proportion=1, flag=wx.ALL | wx.EXPAND)
            buttonBox = wx.BoxSizer(wx.HORIZONTAL)
            for bravais in settings:
                button = wx.ToggleButton(self, -1, system + bravais,
                                         size=(40, 20), style=wx.BU_EXACTFIT)
                button.SetLabel(system + bravais)
                self.Bind(wx.EVT_TOGGLEBUTTON, self.crystemButton, id=button.GetId())
                self.crystemButtons.append(button)
                buttonBox.Add(button, flag=wx.RIGHT | wx.BOTTOM, border=5)
            innerBox.Add(buttonBox)
        csvSizer.Add(innerBox, flag=wx.TOP, border=5)

        cphBox.Add(csvSizer, flag=wx.EXPAND)
        cphBox.Add(lpvSizer, flag=wx.EXPAND)

        ctrlVbox.Add(cphBox, flag=wx.ALL | wx.EXPAND, border=10)

        icphBox = wx.BoxSizer(wx.HORIZONTAL)  # import & cell projection sizer

        importBox = wx.StaticBox(self, -1, "Import a Structure File")
        importVSizer = wx.StaticBoxSizer(importBox, wx.VERTICAL)  # holds the rows
        # importHSizer = wx.BoxSizer(wx.HORIZONTAL)  # holds the columns in a row
        randomText1 = wx.StaticText(self, label="Drag a CIF / POSCAR / geometry.in file below:")
        randomText2 = wx.StaticText(self, label="Choose prototype from sample structures:")

        file_drop_target = MyFileDropTarget(self)
        fileTextCtrl = wx.TextCtrl(self, size=(400, 40),
                                   style=wx.TE_MULTILINE | wx.TE_READONLY)
        fileTextCtrl.SetDropTarget(file_drop_target)

        importVSizer.Add(randomText1, flag=wx.ALL | wx.EXPAND, border=5)
        importVSizer.Add(fileTextCtrl, flag=wx.ALL | wx.EXPAND, border=5)
        importVSizer.Add(randomText2, flag=wx.ALL | wx.EXPAND, border=5)
        # *********************************************************************
        box = wx.BoxSizer(wx.HORIZONTAL)

        box.Add(self.protoSystems, 0, wx.EXPAND)
        box.Add(self.bravaisBox, 1, wx.EXPAND)
        box.Add(self.structBox, 1, wx.EXPAND)
        importVSizer.Add(box, wx.EXPAND)

        cellProjBox = wx.StaticBox(self, -1, "Two Dimensional Cell Projection")
        cellProjVSizer = wx.StaticBoxSizer(cellProjBox, wx.VERTICAL)  # holds the rows

        ccsBox = wx.StaticBox(self, -1, "Crystal System")
        ccsHSizer = wx.StaticBoxSizer(ccsBox, wx.HORIZONTAL)
        ccsText = wx.StaticText(self, label=self.currentCrystem)
        ccsText.SetFont(self._12)
        ccsHSizer.Add(ccsText)

        apdBox = wx.StaticBox(self, -1, "Available Projection Directions")
        apdVSizer = wx.StaticBoxSizer(apdBox, wx.VERTICAL)
        pdText = wx.StaticText(self, label="Calculates 2D parameters a', b', γ' for the"
                                           " projections listed in ITC "
                                           "space-group tables")
        pdText.SetFont(self._12)
        pdText.Wrap(240)
        pstBox = wx.FlexGridSizer(cols=2, rows=4, vgap=0, hgap=5)
        apdVSizer.Add(pdText, flag=wx.ALL, border=5)
        line = wx.StaticLine(self, size=(240, -1), style=wx.LI_HORIZONTAL)
        apdVSizer.Add(line, wx.EXPAND | wx.ALL, 10)
        for row in ControlPanel.symDirs:
            currDir = ControlPanel.symDirs[row]
            dirBtnBox = wx.BoxSizer(wx.HORIZONTAL)
            pstText = wx.StaticText(self, label=row)
            pstText.SetFont(self._10)
            pstBox.Add(pstText, flag=wx.ALL, border=5)
            for col in currDir:
                # pdMethod = getattr(crystemObj.projectionDirections, pd)
                pdBtn = wx.ToggleButton(self, -1, col, size=(40, 20), style=wx.BU_EXACTFIT)
                self.Bind(wx.EVT_TOGGLEBUTTON, self.pdButton, id=pdBtn.GetId())
                # self.crystemButtons.append(button)
                dirBtnBox.Add(pdBtn, flag=wx.RIGHT | wx.BOTTOM, border=5)
            pstBox.Add(dirBtnBox)
            # apdVSizer.Add(pstBox)
        apdVSizer.Add(pstBox, flag=wx.TOP, border=15)

        cellProjVSizer.Add(ccsHSizer, flag=wx.ALL, border=5)
        cellProjVSizer.Add(apdVSizer, flag=wx.ALL, border=5)

        icphBox.Add(cellProjVSizer, flag=wx.ALL | wx.EXPAND, border=5)
        icphBox.Add(importVSizer, flag=wx.ALL | wx.EXPAND, border=5)

        ctrlVbox.Add(icphBox, flag=wx.ALL | wx.EXPAND, border=5)
        self.SetSizer(ctrlVbox)

    @staticmethod
    def adjustName(name):
        for c in name:
            if c.isupper():
                name = name.replace(c, '-' + c.lower())
        return name

    def onCrystemListBox(self, event):
        self.currBravaisOpts = list()
        self.bravaisBox.Clear()
        self.currStructOpts = list()
        self.structBox.Clear()
        index = self.protoSystems.GetSelection()
        symbol = self.protoSystems.GetString(index)
        currDict = self.crystemDict[symbol]
        bravSettings = currDict['settings']
        for brav in bravSettings:
            curr = self.adjustName(bravSettings[brav][0])
            fullName = "{b}\t{f}".format(b=brav, f=curr)
            self.bravaisBox.Append(fullName)

    def onBravaisListBox(self, event):
        self.currStructOpts = list()
        self.structBox.Clear()
        index = self.bravaisBox.GetSelection()
        bravais = self.bravaisBox.GetString(index)[0]
        symbol = self.protoSystems.GetString(
            self.protoSystems.GetSelection())
        protosByFamName = self.getBoxChoices()
        for system in protosByFamName:
            if system == symbol:
                currCrystem = protosByFamName[system]
                print(currCrystem)
                for struct in currCrystem[bravais]:
                    self.structBox.Append(struct)

    def onStructListBox(self, event):
        index = self.structBox.GetSelection()
        structName = self.structBox.GetString(index)
        symbol = self.protoSystems.GetString(
            self.protoSystems.GetSelection())
        bravais = self.bravaisBox.GetString(
            self.bravaisBox.GetSelection())[0]
        structDict = self.prototypes[symbol + bravais]\
            ['structs'][structName]
        crystem = self.crystemDict[symbol]
        bravaisLattices = crystem['settings']
        obj = crystem['instance']
        assert (bravais in bravaisLattices)
        toMF = dict()
        print('structDict[lengths]', structDict['lengths'])
        print('structDict[angles]', structDict['angles'])
        self.a, self.b, self.c = structDict['lengths']
        self.alp, self.bet, self.gam = structDict['angles']
        fxnName_conv, reqParams_conv = bravaisLattices['P']
        method_conv = getattr(obj, fxnName_conv)
        basis_conv = self.calculateBasis(reqParams_conv, method_conv)
        if bravais != 'P':
            fxnName_prim, reqParams_prim = bravaisLattices[bravais]
            method_prim = getattr(obj, fxnName_prim)
        else:
            assert (bravais == 'P')
            fxnName_prim, reqParams_prim = fxnName_conv, reqParams_conv
            method_prim = method_conv
        basis_prim = self.calculateBasis(reqParams_prim, method_prim)
        toMF['name'] = structName
        toMF['dict'] = structDict
        toMF['crystem'] = symbol
        toMF['bravais'] = bravais
        toMF['basis_prim'] = basis_prim
        toMF['basis_conv'] = basis_conv
        pub.sendMessage('apListener', message=toMF,
                        listener='apListener')

    def onEvent(self):
        pass

    def parameterSave(self):
        pass

    def pdButton(self):
        pass

    @staticmethod
    def validityCheck(param):
        if not param:
            return None
        elif all(i.isdigit() or i == '.' for i in param):
            assert (type(float(param)) == float)
            return float(param)
        else:
            print('non-numeric values')
            return None

    def getCrystalSystem(self, a, b, c, alp, bet, gam):
        print("getCrystalSystem")
        epsilon = 10 ** -10
        if abs(b - a) < epsilon:
            # a == b == c --> cubic; trigonal (rhombohedral)
            if abs(c - b) < epsilon:
                # cubic --> alp == bet == gam == pi/2
                if abs(bet - alp) < epsilon and abs(gam - bet) < epsilon \
                        and abs(bet - np.pi / 2.0) < epsilon:
                    return self.crystemDict['c']
                # trigonal --> alp == bet == gam != pi/2
                elif abs(bet - alp) < epsilon < abs(bet - np.pi / 2.0) \
                        and abs(gam - bet) < epsilon:
                    return self.crystemDict['r']
                else:
                    print("None")
                    return None
            else:
                # a == b != c --> tetragonal; hexagonal
                assert (abs(c - b) > epsilon)
                # tetragonal --> alp == bet == gam == pi/2
                if abs(bet - alp) < epsilon and abs(gam - bet) < epsilon \
                        and abs(bet - np.pi / 2.0) < epsilon:
                    return self.crystemDict['t']
                # hexagonal --> alp == bet == pi/2 and gam == 2pi/3
                elif abs(bet - alp) < epsilon and abs(bet - np.pi / 2.0) < epsilon \
                        and abs(gam, 2 * np.pi / 3.0) < epsilon:
                    return self.crystemDict['h']
                else:
                    print("None")
                    return None
        else:
            # a != b != c --> triclinic; monoclinic; orthorhombic
            assert (abs(b - a) > epsilon and abs(c - b) > epsilon)
            # triclinic -->  alp != bet != gam != pi/2
            if abs(bet - alp) > epsilon and abs(gam - bet) > epsilon \
                    and abs(bet - np.pi / 2.0) > epsilon:
                return self.crystemDict['a']
            # monoclinic --> alp == gam == pi/2 != bet and bet != pi/2 != 2pi/3
            elif abs(gam - alp) < epsilon < abs(bet - np.pi / 2.0) < abs(
                    bet - 2 * np.pi / 3.0) and abs(alp - np.pi / 2.0) < epsilon:
                return self.crystemDict['m']
            # orthorhombic --> alp == bet == gam == pi/2
            elif abs(bet - alp) < epsilon and abs(gam - bet) < epsilon \
                    and abs(bet - np.pi / 2.0) < epsilon:
                return self.crystemDict['o']
            else:
                print("None")
                return None

    @staticmethod
    def wrapper(method, args):
        return method(*args)

    def setRequiredConstants(self):
        for i in range(len(self.axialLengths)):
            length = self.axialLengths[i]
            if length.IsEnabled():
                lenInput = length.GetValue()
                if self.validityCheck(lenInput):
                    setattr(self, 'abc'[i], float(lenInput))
                else:
                    wx.MessageBox('Enter an int or float', 'Warning',
                                  wx.OK | wx.ICON_WARNING)
                    return False
        attribs = ['alp', 'bet', 'gam']
        for j in range(len(self.interaxialAngles)):
            angle = self.interaxialAngles[j]
            if angle.IsEnabled():
                angInput = angle.GetValue()
                if self.validityCheck(angInput):
                    setattr(self, attribs[j], float(angInput))
                else:
                    wx.MessageBox('Enter an int or float', 'Warning',
                                  wx.OK | wx.ICON_WARNING)
                    return False
        return True

    def calculateBasis(self, reqParams, basisMethod):
        params = list()
        for i in reqParams:
            params.append(getattr(self, i))
        basis = self.wrapper(basisMethod, params)
        print('basis\t', basis)
        basis_inv = np.linalg.inv(basis)
        print('inverse\t', basis_inv)
        '''
        for i in range(len(basis_inv)):
            for j in range(len(basis_inv[i])):
                if basis_inv[i][j] < 0:
                    basis_inv[i][j] += 1
                elif basis_inv[i][j] > 1:
                    basis_inv[i][j] -= 1
                else:
                    assert(basis_inv[i][j] <= 1)
                    assert(basis_inv[i][j] >= 0)
        print('final\t', basis_inv)
        '''
        # return basis
        return basis

    def disableOtherButtons(self, crystSym):
        for i in self.crystemButtons:
            if i.GetLabel() != crystSym:
                i.Disable()

    def enableOtherButtons(self):
        for i in self.crystemButtons:
            i.Enable()

    # on save button press
    def onSaveStructure(self, event):
        if event.GetEventObject().GetLabel() == 'Save':
            for btn in self.crystemButtons:
                if btn.GetValue():
                    crystSym = btn.GetLabel()
                    family, bravais = crystSym[0], crystSym[1]
                    crystem = self.crystemDict[family]
                    bravaisLattices = crystem['settings']
                    obj = crystem['instance']
                    fxnName_conv, reqParams_conv = bravaisLattices['P']
                    method_conv = getattr(obj, fxnName_conv)
                    if self.setRequiredConstants():
                        basis_conv = self.calculateBasis(reqParams_conv, method_conv)
                        assert (bravais in bravaisLattices)
                        if bravais != 'P':
                            fxnName_prim, reqParams_prim = bravaisLattices[bravais]
                            method_prim = getattr(obj, fxnName_prim)
                        else:
                            assert (bravais == 'P')
                            fxnName_prim, reqParams_prim = fxnName_conv, reqParams_conv
                            method_prim = method_conv
                        basis_prim = self.calculateBasis(reqParams_prim, method_prim)
                        if not self.saveAs.GetValue():
                            saveAs = 'struct_{i}'.format(i=self.structCount + 1)
                            self.saveAs.SetValue(saveAs)
                        else:
                            saveAs = self.saveAs.GetValue()
                        if saveAs not in self.structures:
                            structDict = dict()
                            # basis primitive
                            structDict['basis_prim'] = basis_prim
                            # basis conventional
                            structDict['basis_conv'] = basis_conv
                            structDict['crystem'] = crystem['name']
                            structDict['bravais'] = bravais
                            self.structures[saveAs] = structDict
                    pub.sendMessage('mfListener', message=self.structures, listener='mfListener')

    # noinspection PyUnresolvedReferences
    def crystemButton(self, event):
        btnPressed = event.GetEventObject()
        crystSym = btnPressed.GetLabel()
        family, bravais = crystSym[0], crystSym[1]
        if not btnPressed.GetValue():
            self.enableOtherButtons()
        else:
            self.disableOtherButtons(crystSym)
        if family == 'a':  # TRICLINIC
            pass
        elif family == 'm':  # MONOCLINIC
            alp = self.interaxialAngles[0]
            bet = self.interaxialAngles[2]
            if not event.GetEventObject().GetValue():
                alp.Enable()
                bet.Enable()
            else:
                alp.Disable()
                bet.Disable()
        elif family == 'o':  # ORTHORHOMBIC
            # button is un-toggled; settings reset
            if not event.GetEventObject().GetValue():
                for angleObj in self.interaxialAngles:
                    angleObj.SetValue('')
                    angleObj.Enable()
            else:
                for angleObj in self.interaxialAngles:
                    angleObj.SetValue('90')
                    angleObj.Disable()
        elif family == 't':  # TETRAGONAL
            b = self.axialLengths[1]
            if not event.GetEventObject().GetValue():
                b.Enable()
                for angleObj in self.interaxialAngles:
                    angleObj.SetValue('')
                    angleObj.Enable()
            else:
                b.Disable()
                for angleObj in self.interaxialAngles:
                    angleObj.SetValue('90')
                    angleObj.Disable()
        elif family is 'h':  # HEXAGONAL
            b = self.axialLengths[1]
            if not event.GetEventObject().GetValue():
                b.Enable()
                for angle in self.interaxialAngles:
                    angle.SetValue('')
                    angle.Enable()
            else:
                b.Disable()
                setAngles = ['90', '90', '120']
                for i in range(len(self.interaxialAngles)):
                    angle = self.interaxialAngles[i]
                    angle.SetValue(setAngles[i])
                    angle.Disable()
        elif family is 'r':  # TRIGONAL
            if bravais is 'R':
                if not event.GetEventObject().GetValue():
                    pass
                else:
                    pass
            else:
                assert (bravais is 'P')
        else:  # CUBIC
            assert (family is 'c')
            b, c = self.axialLengths[1:]
            if not event.GetEventObject().GetValue():
                b.Enable()
                c.Enable()
                for angle in self.interaxialAngles:
                    angle.SetValue('')
                    angle.Enable()
            else:
                b.Disable()
                c.Disable()
                setAngles = ['90', '90', '90']
                for i in range(len(self.interaxialAngles)):
                    angle = self.interaxialAngles[i]
                    angle.SetValue(setAngles[i])
                    angle.Disable()

    def onResetTextCtrls(self, event):
        if event.GetEventObject().GetLabel() == 'Reset':
            for i in range(len(self.axialLengths)):
                length = self.axialLengths[i]
                if length.IsEnabled():
                    length.SetValue('')
            for j in range(len(self.interaxialAngles)):
                angle = self.interaxialAngles[j]
                if angle.IsEnabled():
                    angle.SetValue('')

    def onValidateStructure(self, event):
        pass

    def onClear(self, event):
        # clear user input in text control
        pass

    def onGenerateLattice(self, event):
        pass

    def Oncubic(self):
        pub.sendMessage('lp_panelListener',
                        message=dict(lp=self.lattice_parameters,
                                     ra=self.rotation_angle, ax=self.axis,
                                     tv=self.translationVector),
                        listener='lp_panelListener')
