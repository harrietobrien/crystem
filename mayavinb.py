from numpy import sqrt, sin, mgrid
import numpy as np
from crystem import CrystemList
import itertools
from mayavi.core.api import PipelineBase
from traits.api import HasTraits, Instance, Property, \
    List, Array, Int, on_trait_change
from traitsui.api import View, Item, HSplit, \
    VSplit, InstanceEditor, ListEditor
from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.engine_view import EngineView
from mayavi.core.ui.api import MayaviScene
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pubsub import pub
import pandas as pd
import networkx as nx


# class to represent a current structure
class Struct(HasTraits):
    _crystem = None
    _bravais = None
    _struct_name: str = ''
    _ucl_count: int = 0
    _orientation_axes = None
    _prim_matrix, _conv_matrix = None, None
    _ptlist_prim, _ptlist_conv = None, None
    _UC_prim, _UC_conv = None, None
    _BV_prim, _BV_conv = None, None
    _plot_prim, _plot_conv = None, None
    _prim_surf = None
    _scale_factor = 1
    view = View()

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.edges: int = 2
        self.uc_line_names = list()
        self.basis_combinations = None
        pub.subscribe(self.listener, "latticeListener")
        pub.subscribe(self.listener, "ptSizeListener")

    def unitCellReset(self):
        ms = self.app.scene.mlab
        for obj in ms.pipeline.traverse(ms.gcf()):
            if obj.name in self.uc_line_names:
                # print(obj.name)
                pass

    def update(self):
        self.app.scene.update_data()

    def basisVectorsFractional(self):
        return self._prim_matrix, self._conv_matrix

    def basisVectorsCartesian(self):
        x = np.array([1, 0, 0])  # xˆ
        y = np.array([0, 1, 0])  # yˆ
        z = np.array([0, 0, 1])  # zˆ
        pt = np.array([x, y, z])
        inverse = np.linalg.inv(self._crystal_matrix)
        return np.matmul(pt, inverse)

    @staticmethod
    def defineReciprocalLattice():
        # reciprocal lattice defined by b1, b2, b3
        # a_1, a_2, a_3 = self.basisVectors()
        # unit cell volume
        v = np.dot(a_1, np.cross(a_2, a_3))
        c_1 = (2 * np.pi / v)
        c_2 = 2 * np.pi
        b_1 = c_1 * np.cross(a_2, a_3)
        b_2 = c_1 * np.cross(a_3, a_1)
        b_3 = c_1 * np.cross(a_1, a_2)
        u_1 = np.dot(b_1, a_1) / c_2
        u_2 = np.dot(b_2, a_2) / c_2
        u_3 = np.dot(b_3, a_3) / c_2
        return b_1, b_2, b_3

    def extraLatticePoints(self, bravais):
        # A centering -- bc face
        if bravais == 'A':
            return np.array([0, .5, .5])
        # B centering -- ac face
        elif bravais == 'B':
            return np.array([.5, 0, .5])
        # C centering -- ab face
        elif bravais == 'C':
            return np.array([.5, .5, 0])
        # F all centers -- np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
        elif bravais == 'F':  # positioned at (0, 0, 0)
            # a1 --> xy-plane
            a1_pt = self._prim_matrix[:, 0]
            # a2 --> yz-plane
            a2_pt = self._prim_matrix[:, 1]
            # a3 --> xz-plane
            a3_pt = self._prim_matrix[:, 2]
            a = a2_pt + a3_pt
            b = a1_pt + a3_pt
            c = a1_pt + a2_pt
            return a1_pt, a2_pt, a3_pt, a, b, c
        # I body centering -- np.array([.5, .5, .5])
        elif bravais == 'I':
            if self._crystem == 'c':
                a3_pt = self._prim_matrix[:, 2]
                return a3_pt  # center point
            assert (self._crystem == 'o' or self._crystem == 't')
        # R centering -- np.array([0, 0, 0])
        elif bravais == 'R':
            return
        # Q obverse -- np.array([[0, 0, 0], [2/3, 1/3, 1/3], [1/3, 2/3, 2/3]])
        elif bravais == 'Q':
            return
        # S reverse -- np.array([[0, 0, 0], [1/3, 2/3, 1/3], [2/3, 1/3, 2/3]])
        elif bravais == 'S':
            return
        # H hexagonal --  np.array([[0, 0, 0], [2/3, 1/3, 0], [1/3, 2/3, 0]])
        elif bravais == 'H':
            return
        else:
            assert (not bravais)

    def getPointList(self):
        assert (self._prim_matrix is not None)
        assert (self._conv_matrix is not None)
        self._ptlist_conv = self.eight_corners(cellType='conventional')
        self._ptlist_prim = self.eight_corners(cellType='primitive')

    def listener(self, message, listener=None):
        if listener == 'latticeListener':
            for name in message:
                struct = message[name]
                self._struct_name = struct.name
                self._crystem = struct.crystem
                self._bravais = struct.bravais
                self._prim_matrix = struct.matrix_prim
                self._conv_matrix = struct.matrix_conv
                self.getPointList()
                self.redraw()
        elif listener == 'ptSizeListener':
            assert (type(message) == float or
                    type(message) == int)
            self._scale_factor = message
            self.redraw()
        else:
            pass

    def redraw(self):
        if hasattr(self, 'app') and self.app.scene._renderer is not None:
            self.app.scene.disable_render = True
            self.display()
            self.displayOrientationAxes()
            self.getConnections('conventional')
            self.getConnections('primitive')
            self.unitCellReset()
            # self.displayPrimitiveSurface()
            self.app.scene.disable_render = False

    def eight_corners(self, cellType='primitive'):
        primitive, conventional = self.basisVectorsFractional()
        if cellType == 'primitive':
            matrix = primitive
        elif cellType == 'conventional':
            matrix = conventional
        else:
            matrix = cellType
        corners = np.array([[0, 0, 0]])
        basis_list = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
        for i in range(1, 4):
            basis_combinations = list(itertools.combinations(basis_list, i))
            for corner in basis_combinations:
                corner = np.array(corner)
                temp = np.array([0., 0., 0.])
                for array in corner:
                    temp += array.dot(matrix)
                corners = np.append(corners, [temp], axis=0)
        return corners

    def getSurfaceStacks(self):
        surfBounds = self.getSurfaceBounds(self.ptlist_prim)
        assert (len(surfBounds) == 6)
        surfStacks = list()
        for points in surfBounds:
            ptStack = np.empty([1, 3], dtype=float)
            tmp = points[2]
            points[2] = points[3]
            points[3] = tmp
            for point in points:
                point = np.array(point)
                ptStack = np.vstack([ptStack, point])
            surfStacks.append(ptStack)
        assert (len(surfStacks) == 6)
        return surfStacks

    @staticmethod
    def getSurfaceBounds(corners):
        bounds = [[0, 1, 3, 5], [0, 2, 3, 6],
                  [1, 4, 5, 7], [2, 4, 6, 7],
                  [0, 1, 2, 4], [3, 5, 6, 7]]
        boundPoints = list()
        for surf in bounds:
            currSurf = list()
            for ptIndex in surf:
                currSurf.append(corners[ptIndex])
            boundPoints.append(currSurf)
        return boundPoints

    def displayPrimitiveSurface(self):
        stacks = self.getSurfaceStacks()
        print('STACKS\n', stacks)
        ns = self.getSurfaceNormals()
        x0, y0, z0 = 0, 0, 0
        for i in range(len(ns)):
            xc, yc, zc = ns[i]
            if i >= 3:
                x0, y0, z0 = 1, 1, 1
            z = (x0 * zc + y0 * xc + z0 * yc - x * xc - y * yc) / zc
        print('NORMALS\n', ns)

    def getSurfaceNormals(self, cellType='primitive'):
        if cellType == 'primitive':
            matrix = self._prim_matrix
        else:
            assert (cellType == 'conventional')
            matrix = self._conv_matrix
        a1, a2, a3 = matrix[:, 0], matrix[:, 1], matrix[:, 2]
        ri = self.rotoInvert(1, matrix)
        print('ri\t', ri)
        a4, a5, a6 = ri[:, 0], ri[:, 1], ri[:, 2]
        n1 = np.cross(a1, a2)
        n2 = np.cross(a1, a3)
        n3 = np.cross(a2, a3)
        n4 = np.cross(a4, a5)
        n5 = np.cross(a4, a6)
        n6 = np.cross(a5, a6)
        print(n1, n2, n3, n4, n5, n6)
        return n1, n2, n3, n4, n5, n6

    @staticmethod
    def rotoInvert(n, matrix):
        # plot to check these
        phi = 2 * np.pi / n
        xri = np.array([-np.cos(phi), np.sin(phi), 0])
        yri = np.array([-np.sin(phi), -np.cos(phi), 0])
        zri = np.array([0, 0, -1])
        transRI = np.column_stack([xri, yri, zri])
        # check matrix shape
        return np.dot(transRI, matrix)

    def create_3D_lattice(self, edges=None, cellType=None):
        primitive, conventional = self.basisVectorsFractional()
        if not edges:
            edges = self.edges
        if cellType == 'primitive':
            matrix = primitive
        elif cellType == 'conventional':
            matrix = conventional
        else:
            assert (cellType is None)
            matrix = self.point_list_comb
        point_list = np.array([0, 0, 0])
        edge_list = np.array([0, 0, 0])
        current_edge = np.array([0, 0, 0])
        for _ in range(1, edges):
            edge_list = np.vstack([edge_list, np.add(current_edge, matrix[:, 0])])
            point_list = np.vstack([point_list, np.add(current_edge, matrix[:, 0])])
            current_edge = edge_list[-1]
        for edge in edge_list:
            current_point = edge
            for i in range(1, edges):
                point_list = np.vstack([point_list, np.add(current_point, matrix[:, 1])])
                current_point = np.add(edge, matrix[:, 1] * i)
        for point in point_list:
            current_point = point
            for i in range(1, edges):
                point_list = np.vstack([point_list, np.add(current_point, matrix[:, -1])])
                current_point = np.add(point, matrix[:, -1] * i)
        return point_list

    @staticmethod
    def printPointList(point_list):
        for (i, pt) in enumerate(point_list):
            print(i + 1, '\t', pt)

    @on_trait_change('_ptlist_prim, _ptlist_conv, struct.activated')
    def getConnections(self, cellType='primitive'):
        connections = [[0, 1], [0, 2], [0, 3],
                       [1, 4], [1, 5], [2, 4],
                       [2, 6], [3, 5], [3, 6],
                       [4, 7], [5, 7], [6, 7]]
        lattice = self.eight_corners(cellType)
        xs = np.hstack(lattice[:, 0])
        ys = np.hstack(lattice[:, 1])
        zs = np.hstack(lattice[:, 2])
        if cellType == 'primitive':
            if self._UC_prim is None:
                self._UC_prim = self.app.scene.mlab.pipeline.scalar_scatter(xs, ys, zs)
                self._UC_prim.mlab_source.dataset.lines = connections
                self._UC_prim.update()
                lines = self.app.scene.mlab.pipeline.stripper(self._UC_prim)
                self.app.scene.mlab.pipeline.surface(lines, line_width=2)
            else:
                self._UC_prim.mlab_source.reset(x=xs, y=ys, z=zs)
        else:
            assert (cellType == 'conventional')
            if self._UC_conv is None:
                self._UC_conv = self.app.scene.mlab.pipeline.scalar_scatter(xs, ys, zs)
                self._UC_conv.mlab_source.dataset.lines = connections
                self._UC_conv.update()
                lines = self.app.scene.mlab.pipeline.stripper(self._UC_conv)
                self.app.scene.mlab.pipeline.surface(lines, line_width=2)
                self.app.scene.mlab.show()
            else:
                self._UC_conv.mlab_source.reset(x=xs, y=ys, z=zs)

    def getUnitCellConnections(self, cellType='primitive'):
        # 8 corners of the unit cell
        if cellType == 'primitive':
            lattice = self._ptlist_prim
        else:
            assert(cellType == 'conventional')
            lattice = self._ptlist_conv
        # defined connections explicitly
        connections = [[0, 1], [0, 2], [0, 3],
                       [1, 4], [1, 5], [2, 4],
                       [2, 6], [3, 5], [3, 6],
                       [4, 7], [5, 7], [6, 7]]
        lines = list()
        for edge in connections:
            i, j = edge
            currLine = [[], [], []]
            for k in range(3):
                currLine[k].append(lattice[i][k])
                currLine[k].append(lattice[j][k])
            lines.append(currLine)
        return lines

    @staticmethod
    def wrapper(method, args):
        return method.mlab_source.reset(*args)

    @on_trait_change('scene.activated')
    def displayUnitCell(self, cellType='primitive'):
        if cellType == 'primitive':
            _plot_attr = '_UC_prim'
        else:
            assert (cellType == 'conventional')
            _plot_attr = '_UC_conv'
        connections = self.getUnitCellConnections(cellType)
        for line in connections:
            self._ucl_count += 1
            xs, ys, zs = line
            currPlot = getattr(self, _plot_attr)
            if currPlot is None:
                if self._ucl_count < 13:
                    name = "{sn}_UC_line_{n}".format(
                        sn=self._struct_name, n=str(self._ucl_count))
                    self.uc_line_names.append(name)
                    _plot = self.app.scene.mlab.plot3d(
                        xs, ys, zs, name=name, tube_radius=None)  # 0.05)
            else:
                self.app.scene.mlab.clf()
                self.wrapper(currPlot, line)
                # self.unitCellReset()
                # setattr(self, _plot_attr, _plot)
        '''
        if cellType == 'primitive':
            if self._UC_prim is None:
                for line in connections:
                    self._ucp_count += 1
                    xs, ys, zs = line
                    if self._ucp_count < 13:
                        name = "{sn}_UC_line_{n}".format(
                            sn=self._struct_name, n=str(self._ucp_count))
                        self._UC_prim = self.app.scene.mlab.plot3d(
                            xs, ys, zs, name=name, tube_radius=0.05)
            else:
                self._UC_prim.mlab_source = None
        else:
            assert (cellType == 'conventional')
            # connections = self.getUnitCellConnections(cellType)
            if self._UC_conv is None:
                for line in connections:
                    self._ucc_count += 1
                    xs, ys, zs = line
                    if self._ucc_count < 13:
                        name = "{sn}_UC_line_{n}".format(
                            sn=self._struct_name, n=str(self._ucc_count))
                        self._UC_conv = self.app.scene.mlab.plot3d(
                            xs, ys, zs, name=name, tube_radius=0.05)
            else:
                self._UC_conv.mlab_source = None
        '''

    @on_trait_change('_prim_surf, struct.activated')
    def displayPrimitiveSurfaceX(self):
        surfStacks = self.getSurfaceStacks()
        if self._prim_surf is None:
            for surfPts in surfStacks:
                xs = surfPts[:, 0]
                ys = surfPts[:, 1]
                zs = surfPts[:, 2]
                self._prim_surf = self.app.scene.mlab.mesh(xs, ys, zs, warp_scale='auto')
        else:
            self._prim_surf.mlab_source.trait_set()

    @on_trait_change('_BV_prim, _BV_conv, struct.activated')
    def displayBasisVectors(self):
        if not self.basisVectorsFractional():
            return
        primitive, conventional = self.basisVectorsFractional()
        a1_pf, a2_pf, a3_pf = primitive
        a1_cf, a2_cf, a3_cf = conventional
        x0, y0, z0 = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        if self._BV_prim is None:
            self._BV_prim = self.app.scene.mlab.quiver3d(
                x0, y0, z0, a1_pf, a2_pf, a3_pf, line_width=4, mode="2darrow",
                name="Primitive Basis Vectors", scale_mode='vector')
        else:
            self._BV_prim.mlab_source \
                .trait_set(x=x0, y=y0, z=z0, u=a1_pf, v=a2_pf, w=a3_pf)
        orange = [255., 165., 0.]
        for j in range(len(orange)):
            rbg = orange[j]
            orange[j] = rbg / 255.
        if self._BV_conv is None:
            self._BV_conv = self.app.scene.mlab.quiver3d(
                x0, y0, z0, a1_cf, a2_cf, a3_cf, color=tuple(orange), line_width=4,
                mode="2darrow", name="Conventional Basis Vectors", scale_mode='vector')
        else:
            self._BV_conv.mlab_source \
                .trait_set(x=x0, y=y0, z=z0, u=a1_cf, v=a2_cf, w=a3_cf)

    @on_trait_change('struct.activated')
    def displayOrientationAxes(self):
        if self._orientation_axes is None:
            self._orientation_axes = self.app.scene.mlab.orientation_axes()

    # display the lattice in 3D scene
    @on_trait_change('_plot_prim, _plot_conv', 'scene.activated')
    def display(self):
        self.app.scene.disable_render = True
        self.displayBasisVectors()
        # primitive cell
        x_pf = self._ptlist_prim[:, 0]
        y_pf = self._ptlist_prim[:, 1]
        z_pf = self._ptlist_prim[:, 2]
        # conventional cell
        x_cf = self._ptlist_conv[:, 0]
        y_cf = self._ptlist_conv[:, 1]
        z_cf = self._ptlist_conv[:, 2]
        # create scalar for color mapping
        s = np.zeros(self.edges ** 3)
        s.fill(0.25)
        if self._plot_prim is None:
            name = self._struct_name + ' Primitive Points'
            self._plot_prim = self.app.scene.mlab. \
                points3d(x_pf, y_pf, z_pf, s, name=name, colormap='Spectral',
                         resolution=32, scale_factor=self._scale_factor, scale_mode='vector')
        else:
            self._plot_prim.mlab_source.reset(x=x_pf, y=y_pf, z=z_pf)

        if self._plot_conv is None:
            name = self._struct_name + ' Conventional Points'
            self._plot_conv = self.app.scene.mlab. \
                points3d(x_cf, y_cf, z_cf, s, name=name,
                         resolution=32, scale_factor=self._scale_factor, scale_mode='vector')
        else:
            self._plot_conv.mlab_source.reset(x=x_cf, y=y_cf, z=z_cf)
        self.app.scene.disable_render = False


# application object
class MayaviNotebook(HasTraits):
    scene = Instance(MlabSceneModel, (), editor=SceneEditor())
    # mayavi engine view
    engine_view = Instance(EngineView)
    struct = List(Instance(Struct, (), allow_none=False),
                  editor=ListEditor(style='custom'),
                  value=[Struct()])

    # current selection in the engine tree view
    current_selection = Property
    view = View(HSplit(VSplit(Item(name='engine_view',
                                   style='custom',
                                   resizable=True,
                                   show_label=False
                                   ),
                              Item(name='current_selection',
                                   editor=InstanceEditor(),
                                   enabled_when='current_selection is not None',
                                   style='custom',
                                   springy=True,
                                   show_label=False),
                              ),
                       Item(name='scene',
                            editor=SceneEditor(),
                            show_label=False,
                            resizable=True,
                            height=500,
                            width=500),
                       ),
                resizable=True,
                scrollable=True
                )

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.engine_view = EngineView(engine=self.scene.engine)
        self.scene.engine.on_trait_change(self._selection_change, 'current_selection')

    @on_trait_change('scene.activated,struct')
    def init_view(self):
        if self.scene._renderer is not None:
            self.scene.scene_editor.background = (0, 0, 0)
            print(self.struct)
            for struct in self.struct:
                struct.app = self

    def _selection_change(self, old, new):
        self.trait_property_changed('current_selection', old, new)

    def _get_current_selection(self):
        return self.scene.engine.current_selection

    def generate_data_mayavi(self):
        # from mayavi.sources.api import ParametricSurface
        # from mayavi.modules.api import Outline, Surface
        x = self.point_list[:, 0]
        y = self.point_list[:, 1]
        z = self.point_list[:, 2]
        self.scene.engine.plot3d(x, y, z)
        scene = Mayavi_Scene()
        # e.plot3d(x, y, z)
        # s = ParametricSurface()
        # e.add_source(s)
        # e.add_module(Outline())
        # e.add_module(Surface())
