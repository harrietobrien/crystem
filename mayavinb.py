from numpy import sqrt, sin, mgrid
import numpy as np
from crystem import CrystemList
import itertools

from traits.api import HasTraits, Instance, Property, \
    List, Array, Int, CFloat, Enum, on_trait_change
from traitsui.api import View, Item, HSplit, \
    VSplit, InstanceEditor, ListEditor
from tvtk.api import tvtk
from tvtk.pyface.scene_editor import SceneEditor
from mayavi.core.ui.engine_view import EngineView
from mayavi.tools.mlab_scene_model import MlabSceneModel
from pubsub import pub
import pandas as pd
import networkx as nx


# class to represent a current structure
class Struct(HasTraits):
    _crystem = None
    _bravais = None
    _struct_name: str = ''
    _ucp_count: int = 0
    _ucc_count: int = 0
    # crystal matrix (basis)
    _prim_matrix, _conv_matrix = None, None
    # plot unit cell lines
    _UC_prim, _UC_conv = None, None
    # vector scene objects
    _BV_prim, _BV_conv = None, None
    # plot points
    _plot_prim, _plot_conv = None, None
    # primitive cell surfaces
    _prim_surf = None
    _scale_factor = 1
    view = View()

    def __init__(self, **traits):
        HasTraits.__init__(self, **traits)
        self.edges: int = 2
        self.uc_line_names = list()
        self.ptlist_prim = None
        self.ptlist_conv = None
        self.extra_points = None
        pub.subscribe(self.listener, "latticeListener")
        pub.subscribe(self.listener, "ptSizeListener")

    def unitCellReset(self):
        self._ucc_count = 0
        self._UC_conv = None
        self._ucp_count = 0
        self._UC_prim = None
        for obj in self.app.scene.mlab.pipeline \
                .traverse(self.app.scene.mlab.gcf()):
            if obj.name in self.uc_line_names:
                pass

    def update(self):
        self.app.scene.update_pipeline()

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
        print('v', v)
        c_1 = (2 * np.pi / v)
        print('c_1', c_1)
        c_2 = 2 * np.pi
        b_1 = c_1 * np.cross(a_2, a_3)
        print('b_1', b_1)
        b_2 = c_1 * np.cross(a_3, a_1)
        print('b_2', b_2)
        b_3 = c_1 * np.cross(a_1, a_2)
        print('b_3', b_3)
        u_1 = np.dot(b_1, a_1) / c_2
        print('u_1', u_1)
        u_2 = np.dot(b_2, a_2) / c_2
        print('u_2', u_2)
        u_3 = np.dot(b_3, a_3) / c_2
        print('u_3', u_3)
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
        self.ptlist_conv = self.eight_corners(cellType='conventional')
        self.ptlist_prim = self.eight_corners(cellType='primitive')
        print('point list primitive')
        print(self.ptlist_prim)
        # self.extra_points = self.extraLatticePoints(self._bravais)
        # print(self.extra_points)
        points = np.vstack([self.ptlist_prim, self.ptlist_conv])
        # print('points')
        # print(points)
        # print(len(points))
        # self.point_list_comb = np.array(list(set(tuple(p) for p in points)))

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
            self.display()
            self.displayUnitCell('conventional')
            self.displayUnitCell('primitive')
            # self.displayPrimitiveSurface()
            self.unitCellReset()

    def eight_corners(self, cellType=None):
        primitive, conventional = self.basisVectorsFractional()
        if cellType == 'primitive':
            matrix = primitive
        else:
            assert (cellType == 'conventional')
            matrix = conventional
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
        assert(len(surfBounds) == 6)
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
        assert(len(surfStacks) == 6)
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

    def getUnitCellConnections(self, cellType='primitive'):
        # 8 corners of the unit cell
        lattice = self.eight_corners(cellType)
        # defined connections explicitly
        connections = [(0, 1), (0, 2), (0, 3),
                       (1, 4), (1, 5), (2, 4),
                       (2, 6), (3, 5), (3, 6),
                       (4, 7), (5, 7), (6, 7)]
        lines = list()
        for edge in connections:
            i, j = edge
            currLine = [[], [], []]
            for k in range(3):
                currLine[k].append(lattice[i][k])
                currLine[k].append(lattice[j][k])
            line = list()
            for pt in currLine:
                line.append(tuple(pt))
            lines.append(tuple(line))
            print(lines)
        return tuple(lines)

    @on_trait_change('_UC_prim, _UC_conv')
    def displayUnitCell(self, cellType='primitive'):
        if cellType == 'primitive':
            connections = self.getUnitCellConnections('primitive')
            if self._UC_prim is None:
                for line in connections:
                    self._ucp_count += 1
                    xs, ys, zs = line
                    if self._ucp_count < 13:
                        name = "{sn}_UC_line_{n}".format(
                            sn=self._struct_name, n=str(self._ucp_count))
                        # self.uc_line_names.append(name)
                        self._UC_prim = self.app.scene.mlab.plot3d(
                            xs, ys, zs, name=name, tube_radius=0.05)
        else:
            assert (cellType == 'conventional')
            connections = self.getUnitCellConnections('conventional')
            if self._UC_conv is None:
                for line in connections:
                    self._ucc_count += 1
                    xs, ys, zs = line
                    if self._ucc_count < 13:
                        name = "{sn}_UC_line_{n}".format(
                            sn=self._struct_name, n=str(self._ucc_count))
                        # self.uc_line_names.append(name)
                        self._UC_conv = self.app.scene.mlab.plot3d(
                            xs, ys, zs, name=name, tube_radius=0.05)

    @on_trait_change('_prim_surf')
    def displayPrimitiveSurface(self):
        surfStacks = self.getSurfaceStacks()
        if self._prim_surf is None:
            for surfPts in surfStacks:
                xs = surfPts[:, 0]
                ys = surfPts[:, 1]
                zs = surfPts[:, 2]
                self._prim_surf = self.app.scene.mlab.mesh(xs, ys, zs, warp_scale='auto')
        else:
            self._prim_surf.mlab_source.trait_set()

    @on_trait_change('_BV_prim, _BV_conv')
    def displayBasisVectors(self):
        if not self.basisVectorsFractional():
            return
        primitive, conventional = self.basisVectorsFractional()
        a1_pf, a2_pf, a3_pf = primitive
        a1_cf, a2_cf, a3_cf = conventional
        x0, y0, z0 = np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])
        xc, yc, zc = np.array([.5, .5, .5]), np.array([.5, .5, .5]), np.array([.5, .5, .5])
        if self._BV_prim is None:
            self._BV_prim = self.app.scene.mlab.quiver3d(
                x0, y0, z0, a1_pf, a2_pf, a3_pf, line_width=4, mode="2darrow",
                name="F", scale_mode='vector')
        else:
            self._BV_prim.mlab_source \
                .trait_set(x=xc, y=yc, z=zc, u=a1_pf, v=a2_pf, w=a3_pf)
        orange = [255., 165., 0.]
        for j in range(len(orange)):
            rbg = orange[j]
            orange[j] = rbg / 255.
        if self._BV_conv is None:
            self._BV_conv = self.app.scene.mlab.quiver3d(
                x0, y0, z0, a1_cf, a2_cf, a3_cf, color=tuple(orange), line_width=3,
                mode="2darrow", name="C", scale_mode='vector')
        else:
            self._BV_conv.mlab_source \
                .trait_set(x=x0, y=y0, z=z0, u=a1_cf, v=a2_cf, w=a3_cf)

    # display the lattice in 3D scene
    @on_trait_change('_plot_prim, _plot_conv')
    def display(self):
        self.app.scene.disable_render = True
        self.displayBasisVectors()
        # primitive cell
        x_pf = self.ptlist_prim[:, 0]
        y_pf = self.ptlist_prim[:, 1]
        z_pf = self.ptlist_prim[:, 2]
        # conventional cell
        x_cf = self.ptlist_conv[:, 0]
        y_cf = self.ptlist_conv[:, 1]
        z_cf = self.ptlist_conv[:, 2]
        '''
        self.app.scene.mlab.orientation_axes()
        if self._plot_prim is None:
            primDF = pd.DataFrame(self.ptlist_prim, columns=['x', 'y', 'z'])
            primLinks = self.getUnitCellConnections('primitive')
            primG = nx.Graph()
            primG.add_edges_from(primLinks)
            primNodes = dict()
            for i in range(len(primG.nodes())):
                n = primG.nodes()[i]
                xyz = primDF.loc[n]
                node = self.app.scene.mlab.points3d(xyz['x'], xyz['y'], xyz['z'])
                primNodes[i] = node
            primEdges = dict()
            for j in range(len(primG.edges())):
                e = primG.edges()[j]
                xyz = primDF.loc[np.array(e)]
                edge = self.app.scene.mlab.plot3d(xyz['x'], xyz['y'], xyz['z'], tube_radius=1)
                primEdges[j] = edge
        # else:
            # self._plot_prim.mlab_source.reset(x=x_pf, y=y_pf, z=z_pf)

        if self._plot_conv is None:
            convDF = pd.DataFrame(self.ptlist_conv, columns=['x', 'y', 'z'])
            convLinks = self.getUnitCellConnections('conventional')
            convG = nx.OrderedGraph()
            convG.add_edges_from(convLinks)
            convNodes = dict()
            for i, n in enumerate(convG.nodes()):
                xyz = convDF.loc[n]
                n = self.app.scene.mlab.points3d(xyz['x'], xyz['y'], xyz['z'])
                convNodes[i] = n
            primEdges = dict()
            for j, e in enumerate(convG.edges()):
                xyz = convDF.loc[np.array(e)]
                primEdges[j] = self.app.scene.mlab.plot3d(xyz['x'], xyz['y'], xyz['z'], tube_radius=1)
        # else:
            # self._plot_conv.mlab_source.reset(x=x_cf, y=y_cf, z=z_cf)
        self.app.scene.disable_render = False
        '''
        # create scalar for color mapping
        s = np.zeros(self.edges ** 3)
        s.fill(0.25)
        self.app.scene.mlab.orientation_axes()
        if self._plot_prim is None:
            self._plot_prim = self.app.scene.mlab. \
                points3d(x_pf, y_pf, z_pf, name=self._struct_name,  # s, colormap='Spectral',
                         resolution=32, scale_factor=self._scale_factor, scale_mode='vector')
        else:
            self._plot_prim.mlab_source.reset(x=x_pf, y=y_pf, z=z_pf)

        if self._plot_conv is None:
            self._plot_conv = self.app.scene.mlab. \
                points3d(x_cf, y_cf, z_cf, s, name=self._struct_name,
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
        # **************************************
        self.engine_view = EngineView(engine=self.scene.engine)
        # Hook up the current_selection to change when the one in the engine
        # changes.  This is probably unnecessary in Traits3
        self.scene.engine.on_trait_change(self._selection_change, 'current_selection')

    @on_trait_change('scene.activated,struct')
    def init_view(self):
        if self.scene._renderer is not None:
            self.scene.scene_editor.background = (0, 0, 0)
            # self.scene.mlab.pipeline.remove(TubeFactory)
            for struct in self.struct:
                struct.app = self
                struct.display()
                struct.displayUnitCell('conventional')
                struct.displayUnitCell('primitive')
                struct.unitCellReset()
                # struct.displayPrimitiveSurface()

    def _selection_change(self, old, new):
        self.trait_property_changed('current_selection', old, new)

    def _get_current_selection(self):
        return self.scene.engine.current_selection

    def visualize_unit_cell(self, struct):
        self.scene.scene.disable_render = True
        for line in struct._connections:
            struct._uce_count += 1
            xs, ys, zs = line
            if struct._uce_count < 13:
                name = "{sn}_UC_line_{n}".format(sn=struct._struct_name, n=str(struct._uce_count))
                # print(name)
                # print(struct._uce_count)
                struct.uc_line_names.append(name)
                # self.app.scene.mlab.plot3d
                struct._unit_cell = self.scene.mlab.pipeline.plot3d(xs, ys, zs, name=name, tube_radius=0.01,
                                                                    figure=self.app.scene.mayavi_scene)
        self.scene.scene.disable_render = False


'''
    def generate_data_mayavi(self):
        """Shows how you can generate data using mayavi instead of mlab."""
        # from mayavi.sources.api import ParametricSurface
        # from mayavi.modules.api import Outline, Surface
        import mayavi.mlab
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


        
    @staticmethod
    def draw_sphere(xc, yc, zc, r):
        mg = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = r * np.cos(mg[0]) * np.sin(mg[1]) + xc
        y = r * np.sin(mg[0]) * np.sin(mg[1]) + yc
        z = r * np.cos(mg[1]) + zc
        return x, y, z
'''
