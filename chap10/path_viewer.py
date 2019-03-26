"""
mavsimPy: path drawing function
    - Beard & McLain, PUP, 2012
    - Update history:
        1/8/2019 - RWB
"""
import sys
sys.path.append("..")
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
from PyQt5 import QtWidgets
from tools.tools import RotationBody2Vehicle

class path_viewer():
    def __init__(self):
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('Path Viewer')
        #self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        sg = QtWidgets.QDesktopWidget().availableGeometry()
        self.window.setGeometry(sg.width()/2.,0,sg.width()/2.,sg.height())
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        #self.window.setCameraPosition(distance=self.scale, elevation=90, azimuth=0)
        self.window.setCameraPosition(distance=self.scale/2., elevation=5, azimuth=25)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the mav been plotted yet?
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.points, self.meshColors = self._get_mav_points()

    ###################################
    # public functions
    def update(self, path, state):
        """
        Update the drawing of the mav.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        mav_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of mav as a rotation matrix R from body to inertial
        R = RotationBody2Vehicle(state.phi, state.theta, state.psi)
        # rotate and translate points defining mav
        rotated_points = self._rotate_points(self.points, R)
        translated_points = self._translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self._points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            if path.flag=='line':
                straight_line_object = self.straight_line_plot(path)
                self.window.addItem(straight_line_object)  # add straight line to plot
            else:  # path.flag=='orbit
                orbit_object = self.orbit_plot(path)
                self.window.addItem(orbit_object)
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors, # defines mesh colors (Nx1)
                                      drawEdges=False,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the mav location
        #view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        #self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    ###################################
    # private functions
    def _rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points

    def _translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def _get_mav_points(self):
        #points are in NED coordinates
        points = np.genfromtxt ('../chap2/polyvert3.csv', delimiter=",")
        #print(points.shape[0])
        points = points.T
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points = R @ points
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        points = R @ points


        #scale points for better rendering
        scale = 0.1
        points = scale * points
        points[0,:] -= scale*180
        points[2,:] -= scale*40


        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        orange = np.array([1.0, 0.647, 0., 1])
        white_gray = np.array([0.9, 0.9, 0.9, 1])
        dark_gray = np.array([0.3, 0.3, 0.3, 1])
        meshColors = np.empty((587, 3, 4), dtype=np.float32)
        meshColors[0:36] = white_gray # middle of wheels
        meshColors[36:79] = green # parts of right fuselage and wing
        meshColors[79:127] = red # parts of left fuselage and wing
        meshColors[127:200] = dark_gray
        meshColors[200:313] = dark_gray
        meshColors[313:356] = white_gray # propeller
        meshColors[356:378] = white_gray # flanges near tip
        meshColors[378:470] = green
        meshColors[470:560] = red
        meshColors[560:563] = green # underbody patch
        meshColors[563:575] = white_gray # windows
        meshColors[575:] = white_gray # inside chamber near front

        return points, meshColors

    def _points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh2 = np.genfromtxt ('../chap2/polyface3.csv', delimiter=",")
        mesh3 = np.array(list(map(lambda x: list(map(lambda y: points[int(y)], x)), mesh2)))
        return mesh3

    def straight_line_plot(self, path):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + self.scale * path.line_direction.item(0),
                            path.line_origin.item(1) + self.scale * path.line_direction.item(1),
                            path.line_origin.item(2) + self.scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        red = np.array([[1., 0., 0., 1]])
        path_color = np.concatenate((red, red), axis=0)
        object = gl.GLLinePlotItem(pos=points,
                                   color=path_color,
                                   width=2,
                                   antialias=True,
                                   mode='lines')
        return object

    def orbit_plot(self, path):
        N = 100
        red = np.array([[1., 0., 0., 1]])
        theta = 0
        points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                            path.orbit_center.item(1),
                            path.orbit_center.item(2)]])
        path_color = red
        for i in range(0, N):
            theta += 2 * np.pi / N
            new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(theta),
                                   path.orbit_center.item(1) + path.orbit_radius * np.sin(theta),
                                   path.orbit_center.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
            path_color = np.concatenate((path_color, red), axis=0)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = points @ R.T
        object = gl.GLLinePlotItem(pos=points,
                                   color=path_color,
                                   width=2,
                                   antialias=True,
                                   mode='line_strip')
        return object
