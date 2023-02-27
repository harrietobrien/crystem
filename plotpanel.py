import numpy as np
import matplotlib
from crystem import CrystemList

matplotlib.use('WXAgg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
import matplotlib.pyplot as plt
import wx
from pubsub import pub
from matplotlib.gridspec import GridSpec


class PlotPanel(wx.Panel):
    def __init__(self, parent):  # *args, **kw
        wx.Panel.__init__(self, parent, -1, size=(1000, 500))
        self.crystems = CrystemList.getCrystems()

        self.getCrystalMatrix()

        pub.subscribe(self.listener, 'lp_panelListener')
        pub.subscribe(self.listener, 'ra_panelListener')
        pub.subscribe(self.listener, 'ax_panelListener')
        pub.subscribe(self.listener, 'tv_panelListener')

        self.lattice_parameters_split = []
        self.rotation_angle_split = []

        self.crystal_matrix = self.getCrystalMatrix()
        self.values_assigned = False
        self.lattice_parameters_to_use = []
        self.point_list = np.array([[0, 0, 0]])
        self.unit_cell_points = np.array([[0, 0, 0]])
        self.unit_cell_line_pts = np.array([0, 0])
        self.unit_cell_line_pts1 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts2 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts3 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts4 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts5 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts6 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts7 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts8 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts9 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts10 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts11 = np.array([[0, 0, 0], [0, 0, 0]])
        self.unit_cell_line_pts12 = np.array([[0, 0, 0], [0, 0, 0]])
        self.for_plane = []
        self.not_real_system = False
        self.rotation_angle = 0
        self.axis = []
        self.translation_vector = []
        self.tran_vec_ready = False
        self.prime_point_list_x = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_y = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_z = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_uvw = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_x_seitz = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_y_seitz = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_z_seitz = np.array([[0.0, 0.0, 0.0]])
        self.prime_point_list_uvw_seitz = np.array([[0.0, 0.0, 0.0]])
        self.rotate_about_x = False
        self.rotate_about_y = False
        self.rotate_about_z = False
        self.rotate_about_uvw = False
        self.seitz_about_x = False
        self.seitz_about_y = False
        self.seitz_about_z = False
        self.seitz_about_uvw = False

        self.is_cubic = False
        self.is_tetragonal = False
        self.is_orthorhombic = False
        self.is_hexagonal = False
        self.is_monoclinic = False
        self.is_triclinic = False
        self.is_trigonal = False
        self.crystal_system = ''

        font = {'family': 'serif', 'color': 'black', 'weight': 'bold', 'size': 8, }
        self.buttonColor = wx.Colour("#292522")

        self.fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(2, 2)
        self.canvas = FigureCanvas(self, -1, self.fig)
        self.ax1 = self.fig.add_subplot(gs[0, 0], projection='3d')
        self.ax1.set_title("3D Structural Arrangement", fontdict=font)
        self.ax1.tick_params(axis='both', which='major', labelsize=7)
        self.ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')
        self.ax2.set_title("View Symmetry Operations", fontdict=font)
        self.ax2.tick_params(axis='both', which='major', labelsize=7)
        self.ax3 = self.fig.add_subplot(gs[1, 0])
        self.ax3.set_title("2D Structural Arrangement", fontdict=font)
        self.ax3.tick_params(axis='both', which='major', labelsize=7)
        self.ax4 = self.fig.add_subplot(gs[1, 1], projection='3d')
        self.ax4.set_title("Unit Cell", fontdict=font)
        self.ax4.tick_params(axis='both', which='major', labelsize=7)

        vbox = wx.BoxSizer(wx.VERTICAL)
        hboxSizer = wx.BoxSizer(wx.HORIZONTAL)
        toolbar = NavigationToolbar2WxAgg(self.canvas)
        hboxSizer.Add(toolbar, proportion=0.75, flag=wx.TOP, border=20)
        toolbar.Realize()
        vbox.Add(hboxSizer, flag=wx.ALL, border=30)
        vbox.Add(self.canvas, 1, flag=wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, border=30)

        self.SetSizer(vbox)
        self.Fit()

    def toolbar(self):
        return self.toolbar

    def listener(self, message, listener=None):
        if listener == 'lp_panelListener':
            lattice_parameters = message.get('lp')
            for number in lattice_parameters.split(','):
                self.lattice_parameters_split.append(float(number))
            self.values_assigned = True

        if listener == 'ra_panelListener':
            self.rotation_angle = float(message.get('ra'))

        if listener == 'ax_panelListener':
            axis = message.get('ax')
            for number in axis.split(','):
                self.axis.append(float(number))
            self.axis = np.asarray(self.axis)

        if listener == 'tv_panelListener':
            trans_vec = message.get('tv')
            for number in trans_vec.split(','):
                self.translation_vector.append(float(number))
            self.translation_vector = np.asarray(self.translation_vector)
            self.tran_vec_ready = True

    def cartesian_to_lattice(self):
        pass

    def lattice_to_cartesian(self):
        pass

    def setLatticeParameters(self):
        self.a = self.lattice_parameters_to_use[0]
        self.b = self.lattice_parameters_to_use[1]
        self.c = self.lattice_parameters_to_use[2]
        self.alp = self.lattice_parameters_to_use[3]
        self.bet = self.lattice_parameters_to_use[4]
        self.gam = self.lattice_parameters_to_use[5]
        self.edges = self.lattice_parameters_to_use[6]

    def getCrystalMatrix(self):
        if None:
            wx.MessageBox('The crystal system defined by the entered '
                          'lattice parameters does not exist.', 'Warning',
                          wx.OK | wx.ICON_WARNING)
        else:
            return

    def create_3D_lattice(self):
        self.edges = int(self.edges)
        point_list = np.array([0, 0, 0])
        edge_list = np.array([0, 0, 0])
        current_edge = np.array([0, 0, 0])
        for i in range(1, self.edges):
            edge_list = np.vstack([edge_list, np.add(current_edge, self.crystal_matrix[0])])
            point_list = np.vstack([point_list, np.add(current_edge, self.crystal_matrix[0])])
            current_edge = edge_list[-1]
        for edge in edge_list:
            current_point = edge
            for i in range(1, self.edges):
                point_list = np.vstack([point_list, np.add(current_point, self.crystal_matrix[1])])
                current_point = np.add(edge, self.crystal_matrix[1] * i)
        for point in point_list:
            current_point = point
            for i in range(1, self.edges):
                point_list = np.vstack([point_list, np.add(current_point, self.crystal_matrix[2])])
                current_point = np.add(point, self.crystal_matrix[2] * i)
        self.point_list = point_list

    def create_unit_cell(self):
        unit_cell_points = np.array([0, 0, 0])
        unit_cell_points = np.vstack([unit_cell_points, self.crystal_matrix[0]])
        unit_cell_points = np.vstack([unit_cell_points, self.crystal_matrix[1]])
        unit_cell_points = np.vstack([unit_cell_points, self.crystal_matrix[2]])
        unit_cell_points = np.vstack([unit_cell_points,
                                      np.add(self.crystal_matrix[0], self.crystal_matrix[1])])
        unit_cell_points = np.vstack([unit_cell_points,
                                      np.add(self.crystal_matrix[0], self.crystal_matrix[2])])
        unit_cell_points = np.vstack([unit_cell_points,
                                      np.add(self.crystal_matrix[1], self.crystal_matrix[2])])
        unit_cell_points = np.vstack([unit_cell_points,
                                      np.add(np.add(self.crystal_matrix[0],
                                                    self.crystal_matrix[1]), self.crystal_matrix[2])])
        self.unit_cell_points = unit_cell_points

        self.unit_cell_line_pts1 = np.vstack([np.array([0, 0, 0]), self.crystal_matrix[0]])
        self.unit_cell_line_pts2 = np.vstack([np.array([0, 0, 0]), self.crystal_matrix[1]])
        self.unit_cell_line_pts3 = np.vstack([self.crystal_matrix[0],
                                              np.add(self.crystal_matrix[0], self.crystal_matrix[1])])
        self.unit_cell_line_pts4 = np.vstack([self.crystal_matrix[1],
                                              np.add(self.crystal_matrix[0], self.crystal_matrix[1])])
        self.unit_cell_line_pts5 = np.vstack([np.array([0, 0, 0]), self.crystal_matrix[2]])
        self.unit_cell_line_pts6 = np.vstack([self.crystal_matrix[0],
                                              np.add(self.crystal_matrix[0], self.crystal_matrix[2])])
        self.unit_cell_line_pts7 = np.vstack([self.crystal_matrix[1],
                                              np.add(self.crystal_matrix[1], self.crystal_matrix[2])])
        self.unit_cell_line_pts8 = np.vstack([np.add(self.crystal_matrix[0], self.crystal_matrix[1]),
                                              np.add(np.add(self.crystal_matrix[0],
                                                            self.crystal_matrix[1]), self.crystal_matrix[2])])
        self.unit_cell_line_pts9 = np.vstack([self.crystal_matrix[2],
                                              np.add(self.crystal_matrix[0], self.crystal_matrix[2])])
        self.unit_cell_line_pts10 = np.vstack([self.crystal_matrix[2],
                                               np.add(self.crystal_matrix[1], self.crystal_matrix[2])])
        self.unit_cell_line_pts11 = np.vstack([np.add(self.crystal_matrix[0],
                                                      self.crystal_matrix[2]), np.add(np.add(self.crystal_matrix[0],
                                                                                             self.crystal_matrix[1]),
                                                                                      self.crystal_matrix[2])])
        self.unit_cell_line_pts12 = np.vstack([np.add(self.crystal_matrix[1], self.crystal_matrix[2]),
                                               np.add(np.add(self.crystal_matrix[0], self.crystal_matrix[1]),
                                                      self.crystal_matrix[2])])
        self.for_plane = [self.unit_cell_line_pts1, self.unit_cell_line_pts2, self.unit_cell_line_pts3,
                          self.unit_cell_line_pts4]

    def rotate_points_x(self):
        # call functions that generate the point list
        self.getCrystalMatrix()
        self.create_3D_lattice()
        # create a ccw rotation matrix about the x-axis
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(self.rotation_angle), -np.sin(self.rotation_angle)],
                                    [0, np.sin(self.rotation_angle), np.cos(self.rotation_angle)]])
        prime_point_list = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            # reshape new points from (3,1) to (3,) for easy plotting
            p_prime.reshape(3, )
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list = np.vstack([prime_point_list, p_prime])
        self.prime_point_list_x = prime_point_list

    def rotate_points_y(self):
        # call functions that generate the point list
        self.getCrystalMatrix()
        self.create_3D_lattice()
        # create a ccw rotation matrix about the y-axis
        rotation_matrix = np.array([[np.cos(self.rotation_angle), 0, np.sin(self.rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(self.rotation_angle), 0, np.cos(self.rotation_angle)]])
        prime_point_list = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            # reshape new points from (3,1) to (3,) for easy plotting
            p_prime.reshape(3, )
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list = np.vstack([prime_point_list, p_prime])
        self.prime_point_list_y = prime_point_list

    def rotate_points_z(self):
        # call functions that generate the point list
        self.getCrystalMatrix()
        self.create_3D_lattice()
        # create a ccw rotation matrix about the z axis
        rotation_matrix = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                                    [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                                    [0, 0, 1]])
        prime_point_list = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            # reshape new points from (3,1) to (3,) for easy plotting
            p_prime.reshape(3, )
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list = np.vstack([prime_point_list, p_prime])
        self.prime_point_list_z = prime_point_list

    def rotate_points_about_uvw(self):
        # call functions that generate the point list
        self.getCrystalMatrix()
        self.create_3D_lattice()
        # basis
        basis = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0.0],
                          [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0.0],
                          [0.0, 0.0, 1.0]])
        uvw_cart = np.matmul(self.axis, basis)
        a = np.cos(self.rotation_angle / 2)
        b = np.sin(self.rotation_angle / 2) * uvw_cart[0]
        c = np.sin(self.rotation_angle / 2) * uvw_cart[1]
        d = np.sin(self.rotation_angle / 2) * uvw_cart[2]
        D = np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                      [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                      [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]
                      ])
        inverse_basis = np.linalg.inv(basis)
        rotation_matrix = np.linalg.multi_dot([inverse_basis, D, basis])
        prime_point_list = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            p_prime = np.matmul(rotation_matrix, point)
            p_prime = np.round_(p_prime, 3)
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list = np.vstack([prime_point_list, p_prime])
        self.prime_point_list_uvw = prime_point_list

    def create_seitz_about_x(self):
        self.getCrystalMatrix()
        self.create_3D_lattice()
        rotation_matrix = np.array([[1, 0, 0],
                                    [0, np.cos(self.rotation_angle), -np.sin(self.rotation_angle)],
                                    [0, np.sin(self.rotation_angle), np.cos(self.rotation_angle)]])
        seitz_matrix = np.identity(4)
        if self.tran_vec_ready:
            seitz_matrix[:3, 3] = self.translation_vector[:3]
        seitz_matrix[:3, :3] = rotation_matrix
        # move point
        prime_point_list_x_seitz = np.array([0.0, 0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            point = np.concatenate((point, np.array([1])), axis=0)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0, 0.0])):
                prime_point_list_x_seitz = np.vstack([prime_point_list_x_seitz, p_prime])
        for p_point in prime_point_list_x_seitz:
            p_point = np.delete(p_point, -1)
            if not np.array_equal(p_point, np.array([0.0, 0.0, 0.0])):
                prime_point_list_x_seitz = np.vstack([prime_point_list_x_seitz, p_point])
        self.prime_point_list_x_seitz = prime_point_list_x_seitz

    def create_seitz_about_y(self):
        self.getCrystalMatrix()
        self.create_3D_lattice()
        rotation_matrix = np.array([[np.cos(self.rotation_angle), 0, np.sin(self.rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(self.rotation_angle), 0, np.cos(self.rotation_angle)]])
        seitz_matrix = np.identity(4)
        seitz_matrix[:3, 3] = self.translation_vector[:3]
        seitz_matrix[:3, :3] = rotation_matrix
        # move point
        prime_point_list_y_seitz = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            point = np.concatenate((point, [1]), axis=0)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            p_prime = np.delete(p_prime, -1)
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list_y_seitz = np.vstack([prime_point_list_y_seitz, p_prime])
        self.prime_point_list_y_seitz = prime_point_list_y_seitz

    def create_seitz_about_z(self):
        self.getCrystalMatrix()
        self.create_3D_lattice()
        rotation_matrix = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0],
                                    [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0],
                                    [0, 0, 1]])
        seitz_matrix = np.identity(4)
        if self.translation_vector:
            seitz_matrix[:3, 3] = self.translation_vector[:3]
        seitz_matrix[:3, :3] = rotation_matrix
        # move point
        prime_point_list_z_seitz = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            point = np.concatenate((point, [1]), axis=0)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            p_prime = np.delete(p_prime, -1)
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list_z_seitz = np.vstack([prime_point_list_z_seitz, p_prime])
        self.prime_point_list_z_seitz = prime_point_list_z_seitz

    def create_seitz_about_uvw(self):
        # call functions that generate the point list
        self.getCrystalMatrix()
        self.create_3D_lattice()
        # basis
        basis = np.array([[np.cos(self.rotation_angle), -np.sin(self.rotation_angle), 0.0],
                          [np.sin(self.rotation_angle), np.cos(self.rotation_angle), 0.0],
                          [0.0, 0.0, 1.0]])
        uvw_cart = np.matmul(self.axis, basis)
        a = np.cos(self.rotation_angle / 2)
        b = np.sin(self.rotation_angle / 2) * uvw_cart[0]
        c = np.sin(self.rotation_angle / 2) * uvw_cart[1]
        d = np.sin(self.rotation_angle / 2) * uvw_cart[2]
        D = np.array([[a ** 2 + b ** 2 - c ** 2 - d ** 2, 2 * b * c - 2 * a * d, 2 * b * d + 2 * a * c],
                      [2 * b * c + 2 * a * d, a ** 2 - b ** 2 + c ** 2 - d ** 2, 2 * c * d - 2 * a * b],
                      [2 * b * d - 2 * a * c, 2 * c * d + 2 * a * b, a ** 2 - b ** 2 - c ** 2 + d ** 2]
                      ])
        inverse_basis = np.linalg.inv(basis)
        rotation_matrix = np.linalg.multi_dot([inverse_basis, D, basis])
        prime_point_list_uvw_seitz = np.array([0.0, 0.0, 0.0])
        for point in self.point_list:
            # reshape points from (3,) to (3,1) for matrix multiplication
            point.reshape(3, 1)
            p_prime = np.matmul(rotation_matrix, point)
            # round so point decimals are not heinous
            p_prime = np.round_(p_prime, 3)
            # reshape new points from (3,1) to (3,) for easy plotting
            p_prime.reshape(3, )
            # add rotated points to point list
            if not np.array_equal(p_prime, np.array([0.0, 0.0, 0.0])):
                prime_point_list_uvw_seitz = np.vstack([prime_point_list_uvw_seitz, p_prime])
        self.prime_point_list_uvw_seitz = prime_point_list_uvw_seitz

    @staticmethod
    def draw_sphere(xc, yc, zc, r):
        mg = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
        x = r * np.cos(mg[0]) * np.sin(mg[1]) + xc
        y = r * np.sin(mg[0]) * np.sin(mg[1]) + yc
        z = r * np.cos(mg[1]) + zc
        return x, y, z

    def Onplot_sym(self):
        self.getCrystalMatrix()
        self.create_3D_lattice()
        self.create_unit_cell()

        x = self.point_list[:, 0]
        y = self.point_list[:, 1]
        z = self.point_list[:, 2]
        self.ax2.scatter(x, y, z, color='gold', edgecolors='black')

        if self.rotate_about_x:
            self.rotate_points_x()
            xr = self.prime_point_list_x[:, 0]
            yr = self.prime_point_list_x[:, 1]
            zr = self.prime_point_list_x[:, 2]
            self.ax2.scatter(xr, yr, zr, color='orange', edgecolors='black')

        if self.rotate_about_y:
            self.rotate_points_y()
            xr = self.prime_point_list_y[:, 0]
            yr = self.prime_point_list_y[:, 1]
            zr = self.prime_point_list_y[:, 2]
            self.ax2.scatter(xr, yr, zr, color='darkmagenta', edgecolors='black')

        if self.rotate_about_z:
            self.rotate_points_z()
            xr = self.prime_point_list_z[:, 0]
            yr = self.prime_point_list_z[:, 1]
            zr = self.prime_point_list_z[:, 2]
            self.ax2.scatter(xr, yr, zr, color='c', edgecolors='black')

        if self.rotate_about_uvw:
            self.rotate_points_about_uvw()
            xr = self.prime_point_list_uvw[:, 0]
            yr = self.prime_point_list_uvw[:, 1]
            zr = self.prime_point_list_uvw[:, 2]
            self.ax2.scatter(xr, yr, zr, color='fuchsia', edgecolors='black')

        if self.seitz_about_x and self.translation_vector != []:
            self.create_seitz_about_x()
            xr = self.prime_point_list_x_seitz[:, 0]
            yr = self.prime_point_list_x_seitz[:, 1]
            zr = self.prime_point_list_x_seitz[:, 2]
            self.ax2.scatter(xr, yr, zr, color='cyan', edgecolors='black')

        if self.seitz_about_y and self.translation_vector != []:
            self.create_seitz_about_y()
            xr = self.prime_point_list_y_seitz[:, 0]
            yr = self.prime_point_list_y_seitz[:, 1]
            zr = self.prime_point_list_y_seitz[:, 2]
            self.ax2.scatter(xr, yr, zr, color='red', edgecolors='black')

        if self.seitz_about_z and self.translation_vector != []:
            self.create_seitz_about_z()
            xr = self.prime_point_list_z_seitz[:, 0]
            yr = self.prime_point_list_z_seitz[:, 1]
            zr = self.prime_point_list_z_seitz[:, 2]
            self.ax2.scatter(xr, yr, zr, color='blue', edgecolors='black')

        if self.seitz_about_uvw and self.translation_vector != []:
            self.create_seitz_about_uvw()
            xr = self.prime_point_list_uvw_seitz[:, 0]
            yr = self.prime_point_list_uvw_seitz[:, 1]
            zr = self.prime_point_list_uvw_seitz[:, 2]
            self.ax2.scatter(xr, yr, zr, color='green', edgecolors='black')

        self.toolbar.update()
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()

    def Onclear_plot_sym(self):
        self.ax4.cla()
        self.toolbar.update()
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()

    def OnPlot(self):

        self.getCrystalMatrix()
        self.create_3D_lattice()
        self.create_unit_cell()

        print(self.translation_vector)
        print(self.axis)

        if self.is_cubic:
            self.crystal_system = 'Cubic (Isometric)'
        elif self.is_trigonal:
            self.crystal_system = 'Trigonal'
        elif self.is_triclinic:
            self.crystal_system = 'Triclinic'
        elif self.is_monoclinic:
            self.crystal_system = 'Monoclinic'
        elif self.is_hexagonal:
            self.crystal_system = 'Hexagonal'
        elif self.is_tetragonal:
            self.crystal_system = 'Tetragonal'
        elif self.is_orthorhombic:
            self.crystal_system = 'Orthorhombic'

        wx.StaticText(self, label='Crystal System: %s' % self.crystal_system, pos=(580, 580))

        x = self.point_list[:, 0]
        y = self.point_list[:, 1]
        z = self.point_list[:, 2]
        t = np.arange(self.edges ** 3)

        x_uc = self.unit_cell_points[:, 0]
        y_uc = self.unit_cell_points[:, 1]
        z_uc = self.unit_cell_points[:, 2]
        r1 = np.zeros(self.edges ** 3)
        r2 = np.zeros(self.edges ** 3)
        r1.fill(0.10)
        r2.fill(0.25)

        for (xi, yi, zi, ri) in zip(x_uc, y_uc, z_uc, r1):
            (xs, ys, zs) = self.draw_sphere(xi, yi, zi, ri)
            self.ax4.plot_surface(xs, ys, zs, color="c", alpha=1)

        for (xi, yi, zi, ri) in zip(x_uc, y_uc, z_uc, r2):
            (xs, ys, zs) = self.draw_sphere(xi, yi, zi, ri)
            self.ax4.plot_surface(xs, ys, zs, color="yellowgreen", alpha=0.4)

        x1 = self.unit_cell_line_pts1[:, 0]
        y1 = self.unit_cell_line_pts1[:, 1]
        z1 = self.unit_cell_line_pts1[:, 2]

        x2 = self.unit_cell_line_pts2[:, 0]
        y2 = self.unit_cell_line_pts2[:, 1]
        z2 = self.unit_cell_line_pts2[:, 2]

        x3 = self.unit_cell_line_pts3[:, 0]
        y3 = self.unit_cell_line_pts3[:, 1]
        z3 = self.unit_cell_line_pts3[:, 2]

        x4 = self.unit_cell_line_pts4[:, 0]
        y4 = self.unit_cell_line_pts4[:, 1]
        z4 = self.unit_cell_line_pts4[:, 2]

        x5 = self.unit_cell_line_pts5[:, 0]
        y5 = self.unit_cell_line_pts5[:, 1]
        z5 = self.unit_cell_line_pts5[:, 2]

        x6 = self.unit_cell_line_pts6[:, 0]
        y6 = self.unit_cell_line_pts6[:, 1]
        z6 = self.unit_cell_line_pts6[:, 2]

        x7 = self.unit_cell_line_pts7[:, 0]
        y7 = self.unit_cell_line_pts7[:, 1]
        z7 = self.unit_cell_line_pts7[:, 2]

        x8 = self.unit_cell_line_pts8[:, 0]
        y8 = self.unit_cell_line_pts8[:, 1]
        z8 = self.unit_cell_line_pts8[:, 2]

        x9 = self.unit_cell_line_pts9[:, 0]
        y9 = self.unit_cell_line_pts9[:, 1]
        z9 = self.unit_cell_line_pts9[:, 2]

        x10 = self.unit_cell_line_pts10[:, 0]
        y10 = self.unit_cell_line_pts10[:, 1]
        z10 = self.unit_cell_line_pts10[:, 2]

        x11 = self.unit_cell_line_pts11[:, 0]
        y11 = self.unit_cell_line_pts11[:, 1]
        z11 = self.unit_cell_line_pts11[:, 2]

        x12 = self.unit_cell_line_pts12[:, 0]
        y12 = self.unit_cell_line_pts12[:, 1]
        z12 = self.unit_cell_line_pts12[:, 2]

        self.ax4.plot(x1, y1, z1, x2, y2, z2, color='crimson', linewidth=2.0)
        self.ax4.plot(x3, y3, z3, x4, y4, z4, color='crimson', linewidth=2.0)
        self.ax4.plot(x5, y5, z5, color='crimson', linewidth=2.0)
        self.ax4.plot(x6, y6, z6, color='crimson', linewidth=2.0)
        self.ax4.plot(x7, y7, z7, color='crimson', linewidth=2.0)
        self.ax4.plot(x8, y8, z8, color='crimson', linewidth=2.0)
        self.ax4.plot(x9, y9, z9, color='crimson', linewidth=2.0)
        self.ax4.plot(x10, y10, z10, color='crimson', linewidth=2.0)
        self.ax4.plot(x11, y11, z11, color='crimson', linewidth=2.0)
        self.ax4.plot(x12, y12, z12, color='crimson', linewidth=2.0)

        self.ax1.scatter(x, y, z, alpha=0.8, c=t, cmap='cool', edgecolors='', s=60)
        self.ax3.scatter(x, y, alpha=0.8, c=t, cmap='cool', edgecolors='', s=60)

        self.toolbar.update()
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()

    def OnClearPlot(self):
        self.ax1.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.toolbar.update()
        self.fig.set_canvas(self.canvas)
        self.canvas.draw()

