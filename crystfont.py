import sys
import warnings

import logger
import wx

sys.path.append('data/cryst.ttf')


class TestPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        wx.Font.AddPrivateFont('data/cryst.ttf')

        if hasattr(wx.Font, "CanUsePrivateFont") and wx.Font.CanUsePrivateFont():
            wx.Font.AddPrivateFont('data/cryst.ttf')
            print('yay')
        else:
            warnings.warn("font not supported")




if '__name__' == '__main__':
    test = TestPanel()
