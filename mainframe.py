import numpy as np
import wx
import wx.aui
import wx.richtext as rt
from pubsub import pub
from dataclasses import dataclass
from typing import Dict, Optional

from mayavinb import MayaviNotebook
from ctrlpanel import ControlPanel
from plotpanel import PlotPanel


class BKGDPanel(wx.Panel):
    def __init__(self, *a, **k):
        wx.Panel.__init__(self, *a, **k)
        bkgd = wx.Colour("#262322")
        self.SetBackgroundColour(bkgd)


STRUCT_LIST = dict()


@dataclass
class Structure:
    name: str
    crystem: str
    bravais: str
    matrix_prim: np.array
    matrix_conv: np.array
    lengths: list = None
    angles: list = None
    noAtoms: int = None
    noSpecies: int = None
    sites: list = None
    hm: str = None
    csg: str = None
    hallNo: int = None
    dict: Optional[Dict] = None

    def __init__(self):
        print()

    @staticmethod
    def printMatrix(self) -> str:
        return np.array2string(self.matrix_prim)


class MainFrame(wx.Frame):
    BTYPES = {'P': 'primitive',
              'C': 'base-centered',
              'I': 'body-centered',
              'F': 'face-centered',
              'R': 'rhombohedral'}

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, title="Crystem", size=(1600, 1000))
        bkgd = wx.Colour("#262322")
        self.SetBackgroundColour(bkgd)
        self.createMenuBar()

        pub.subscribe(self.mainframeListener, "mfListener", )
        pub.subscribe(self.mainframeListener, "apListener")

        self.mfSizer = wx.BoxSizer(wx.HORIZONTAL)  # main frame sizer
        self.nbSizer = wx.BoxSizer(wx.VERTICAL)  # notebook sizer
        anb = wx.aui.AuiNotebook(self, id=-1, pos=wx.DefaultPosition, size=(800, 1000))

        mayaView = MayaviNotebook()  # add mayavi notebook
        mayaPanel = mayaView.edit_traits(parent=self, kind='subpanel').control
        anb.AddPage(mayaPanel, 'Mayavi')

        columns = [('Name', 100), ('Crystal System', 100),
                   ('Bravais Type', 100), ('Basis', 500)]

        self.structList = wx.ListCtrl(self, -1, size=(800, 100), style=wx.LC_REPORT)
        for k in enumerate(columns):
            index, column = k
            name, width = column[0], column[1]
            self.structList.InsertColumn(index, name, width=width)
        self.structList.Bind(wx.EVT_LIST_ITEM_SELECTED,
                             self.listItemPressed, self.structList)
        self.nbSizer.Add(anb, 1, wx.ALL | wx.EXPAND, 5)

        savedStructBox = wx.StaticBox(self, -1, "Saved Structures")
        ssBoxSizer = wx.StaticBoxSizer(savedStructBox, wx.HORIZONTAL)

        ssBoxSizer.Add(self.structList)
        self.nbSizer.Add(ssBoxSizer, flag=wx.ALL, border=10)
        # symmetry operation / current structure sizer
        self.hSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.currentStructure()  # adds currStruct to hSizer
        self.symmetryOperations()  # adds symop to hSizer
        self.nbSizer.Add(self.hSizer, flag=wx.LEFT | wx.TOP | wx.BOTTOM, border=5)
        self.mfSizer.Add(self.nbSizer, flag=wx.LEFT | wx.TOP | wx.BOTTOM, border=5)

        ctrlPanel = ControlPanel(self)
        self.mfSizer.Add(ctrlPanel, flag=wx.RIGHT | wx.TOP | wx.BOTTOM, border=5)

        self.SetAutoLayout(True)
        self.SetSizer(self.mfSizer)
        self.Layout()

    def currentStructure(self):
        cstructBox = wx.StaticBox(self, -1, 'Current Structure Details')
        cstructSizer = wx.StaticBoxSizer(cstructBox, wx.VERTICAL)
        self.outRTC = rt.RichTextCtrl(self, size=(320, 280), style=wx.TE_MULTILINE |
                                                                   wx.SUNKEN_BORDER | wx.VSCROLL | wx.HSCROLL)
        cstructSizer.Add(wx.StaticText(self, label='STRUCTURE'), flag=wx.LEFT, border=10)
        cstructSizer.Add(self.outRTC, flag=wx.LEFT | wx.EXPAND, border=10)
        self.hSizer.Add(cstructSizer, flag=wx.ALL | wx.EXPAND, border=5)

    def symmetryOperations(self):
        symopBox = wx.StaticBox(self, -1, 'Symmetry Operations')
        symopSizer = wx.StaticBoxSizer(symopBox, wx.VERTICAL)
        raText = wx.StaticText(self, label='Rotation Angle(s)')
        symopSizer.Add(raText, flag=wx.LEFT, border=10)

        ss = ControlPanel.subscripts
        angleBox = wx.BoxSizer(wx.HORIZONTAL)
        thetaChoices = ['\u03B8' + ss[i] for i in range(len(ss))]
        for theta in thetaChoices:
            thetaText = wx.StaticText(self, label=theta)
            angleBox.Add(thetaText, proportion=0.5, flag=wx.ALL, border=5)
            rotAngleCtrl = wx.SpinCtrl(self, value='1', style=wx.SP_VERTICAL)
            rotAngleCtrl.SetRange(1, 360)
            angleBox.Add(rotAngleCtrl, proportion=1, flag=wx.ALL, border=5)

        rotButton = wx.Button(self, -1, 'Enter', style=wx.BU_EXACTFIT)
        self.Bind(wx.EVT_SPIN, self.onEvent, id=rotButton.GetId())
        angleBox.Add(rotButton, flag=wx.ALL, border=5)
        symopSizer.Add(angleBox, flag=wx.ALL | wx.EXPAND, border=5)
        uvwText = wx.StaticText(self, label="Rotation Axis [uvw]")
        uvwBox = wx.BoxSizer(wx.HORIZONTAL)
        symopSizer.Add(uvwText, flag=wx.LEFT, border=5)
        symops = list('uvwuuu')

        rotAxis = symops[0:len(symops) // 2]
        trxVect = symops[len(symops) // 2:]

        for i in rotAxis:
            text = wx.StaticText(self, label=i)
            ctrl = wx.TextCtrl(self, value='', style=wx.TE_PROCESS_ENTER)
            uvwBox.Add(text, proportion=0.5, flag=wx.ALL, border=5)
            uvwBox.Add(ctrl, proportion=1, flag=wx.ALL, border=5)
        uvwButton = wx.Button(self, -1, 'Enter', style=wx.BU_EXACTFIT)
        self.Bind(wx.EVT_BUTTON, self.onEvent, id=uvwButton.GetId())
        uvwBox.Add(uvwButton, flag=wx.ALL, border=5)
        symopSizer.Add(uvwBox, flag=wx.ALL | wx.EXPAND, border=5)

        tvText = wx.StaticText(self, label=f"Translation Vector "
                                           f"[u{ss[0]} u{ss[1]} u{ss[2]}]")
        symopSizer.Add(tvText, flag=wx.LEFT, border=5)
        tvSizer = wx.BoxSizer(wx.HORIZONTAL)
        for i in range(len(trxVect)):
            text = wx.StaticText(self, label=trxVect[i] + ss[i])
            ctrl = wx.TextCtrl(self, value='', style=wx.TE_PROCESS_ENTER)
            tvSizer.Add(text, proportion=0.5, flag=wx.ALL, border=5)
            tvSizer.Add(ctrl, proportion=1, flag=wx.ALL, border=5)
        tvButton = wx.Button(self, -1, 'Enter', style=wx.BU_EXACTFIT)
        self.Bind(wx.EVT_BUTTON, self.onEvent, id=tvButton.GetId())
        tvSizer.Add(tvButton, flag=wx.ALL, border=5)
        symopSizer.Add(tvSizer, flag=wx.ALL | wx.EXPAND, border=5)
        symopSizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.TOP | wx.BOTTOM, 5)
        newText = wx.StaticText(self, label="Plot Configuration")
        symopSizer.Add(newText, flag=wx.ALL | wx.EXPAND, border=10)

        buttonSizer = wx.BoxSizer(wx.VERTICAL)
        axes = [axis + '-axis' for axis in 'x y z uvw'.split()]

        ccwCheckBox = wx.CheckBox(self, label='A')
        ccwCheckBox.Bind(wx.EVT_CHECKBOX, self.onEvent)
        ccwThetaCB = wx.ComboBox(self, -1, choices=thetaChoices)
        ccwThetaCB.Bind(wx.EVT_COMBOBOX, self.onEvent)
        ccwAxesCB = wx.ComboBox(self, -1, choices=axes)
        ccwAxesCB.Bind(wx.EVT_COMBOBOX, self.onEvent)
        ccwComps = [ccwCheckBox, wx.StaticText(self, label='CCW rotation of'),
                    ccwThetaCB, wx.StaticText(self, label='about the'), ccwAxesCB]
        ccwSizer = wx.BoxSizer(wx.HORIZONTAL)
        for component in ccwComps:
            ccwSizer.Add(component, flag=wx.ALL | wx.EXPAND, border=10)

        seitzCheckBox = wx.CheckBox(self, label='B')
        seitzCheckBox.Bind(wx.EVT_CHECKBOX, self.onEvent)
        seitzLabel = 'Seitz operation with rotation about the'
        seitzText = wx.StaticText(self, label=seitzLabel)
        seitzAxesCB = wx.ComboBox(self, -1, choices=axes)
        seitzComps = [seitzCheckBox, seitzText, seitzAxesCB]
        seitzSizer = wx.BoxSizer(wx.HORIZONTAL)
        for component in seitzComps:
            seitzSizer.Add(component, flag=wx.ALL | wx.EXPAND, border=10)

        buttonSizer.Add(ccwSizer, proportion=1)
        buttonSizer.Add(seitzSizer, proportion=1)
        symopSizer.Add(buttonSizer, flag=wx.EXPAND)

        self.hSizer.Add(symopSizer, flag=wx.ALL, border=5)

    def onEvent(self):
        pass

    def listItemPressed(self, event):
        index = event.GetIndex()
        item = self.structList.GetItem(index, 0)
        name = item.GetText()
        struct: Structure = STRUCT_LIST[name]
        # put stuff in structure details
        if struct.dict:
            a, b, c = struct.dict['lengths']
            alp, bet, gam = struct.dict['angles']
            out = 'a\t{a}\nb\t{b}\nc\t{c}\n' \
                  'α\t{alp}\nβ\t{bet}\nγ\t{gam}\n' \
                .format(atoms=struct.dict['#atoms'],
                        species=struct.dict['#species'],
                        a=a, b=b, c=c,
                        alp=alp, bet=bet, gam=gam)
            self.outRTC.WriteText(out)
            self.outRTC.Newline()
        pub.sendMessage('latticeListener',
                        message=dict(name=struct),
                        listener='latticeListener')

    def inListCtrl(self, struct):
        item = self.structList.FindItem(-1, struct)
        if item == -1:
            return False
        return True

    # populates 'Saved Structures' list
    def populateListCtrl(self):
        for structTuple in enumerate(STRUCT_LIST):
            index, struct = structTuple
            if not self.inListCtrl(struct):
                structObj = STRUCT_LIST[struct]
                basis: str = structObj.printMatrix(structObj)
                crystem: str = structObj.crystem
                self.structList.InsertItem(index, struct)
                self.structList.SetItem(index, 1, crystem)
                self.structList.SetItem(index, 2, structObj.bravais)
                self.structList.SetItem(index, 3, basis)

    def mainframeListener(self, message, listener=None):
        if listener == 'mfListener':
            for struct in message:
                structDict = message[struct]
                newStruct = Structure()
                newStruct.name = struct
                newStruct.crystem = structDict['crystem']
                bravais = structDict['bravais']
                newStruct.bravais = MainFrame.BTYPES[bravais]
                newStruct.matrix_prim = structDict['basis_prim']
                newStruct.matrix_conv = structDict['basis_conv']
                STRUCT_LIST[struct] = newStruct
        elif listener == 'apListener':
            newStruct = Structure()
            newStruct.name = message['name']
            newStruct.dict = message['dict']
            newStruct.crystem = message['crystem']
            newStruct.bravais = message['bravais']
            newStruct.matrix_prim = message['basis_prim']
            newStruct.matrix_conv = message['basis_conv']
            STRUCT_LIST[message['name']] = newStruct
        self.populateListCtrl()

    def addPrototypeListener(self, message, listener=None):
        pass

    def createMenuBar(self):
        menubar = wx.MenuBar()
        menu1 = wx.Menu()
        menubar.Append(menu1, "References")
        self.SetMenuBar(menubar)


if __name__ == '__main__':
    app = wx.App(False)
    fr = wx.Frame(None, title='test')
    frame = MainFrame(None)
    frame.Show(True)
    app.MainLoop()
