"""
example of drawing a box-like spacecraft in python
    - Beard & McLain, PUP, 2012
    - Update history:
        1/8/2019 - RWB
"""
import numpy as np
from stl import mesh as mesh_mod
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector

class mav_viewer():
    def __init__(self):
        # initialize Qt gui application and window
        pg.setConfigOptions(antialias=True)
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('EC EN 674: Flight Dynamics & Controls Design Project')
        self.window.showMaximized()
        #self.window.setGeometry(0, 0, 1000, 1000)  # args: upper_left_x, upper_right_y, width, height
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(20, 20, 5) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=200) # distance from center of plot to camera
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the spacecraft been plotted yet?
        # get points that define the non-rotated, non-translated spacecraft and the mesh colors
        self.points, self.meshColors = self._get_spacecraft_points()

    ###################################
    # public functions
    def update(self, state):
        """
        Update the drawing of the spacecraft.

        The input to this function is a (message) class with properties that define the state.
        The following properties are assumed to be:
            state.pn  # north position
            state.pe  # east position
            state.h   # altitude
            state.phi  # roll angle
            state.theta  # pitch angle
            state.psi  # yaw angle
        """
        spacecraft_position = np.array([[state.pn], [state.pe], [-state.h]])  # NED coordinates
        # attitude of spacecraft as a rotation matrix R from body to inertial
        R = self._Euler2Rotation(state.phi, state.theta, state.psi)
        # rotate and translate points defining spacecraft
        rotated_points = self._rotate_points(self.points, R)
        translated_points = self._translate_points(rotated_points, spacecraft_position)
        # convert North-East Down to East-North-Up for rendering
        R_disp = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        translated_points = R_disp @ translated_points
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self._points_to_mesh(translated_points)

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                      vertexColors=self.meshColors, # defines mesh colors (Nx1)
                                      drawEdges=True,  # draw edges between mesh elements
                                      smooth=False,  # speeds up rendering
                                      computeNormals=False)  # speeds up rendering
            self.window.addItem(self.body)  # add body to plot
            # add all three axis
            axis_length = 220.0
            naxis_pts = np.array([[0.0,0.0,0.0],
                            [0.0,axis_length,0.0]])
            naxis = gl.GLLinePlotItem(pos=naxis_pts,color=pg.glColor('r'),width=3.0)
            self.window.addItem(naxis)
            eaxis_pts = np.array([[0.0,0.0,0.0],
                            [axis_length,0.0,0.0]])
            eaxis = gl.GLLinePlotItem(pos=eaxis_pts,color=pg.glColor('g'),width=3.0)
            self.window.addItem(eaxis)
            daxis_pts = np.array([[0.0,0.0,0.0],
                            [0.0,0.0,-axis_length]])
            daxis = gl.GLLinePlotItem(pos=daxis_pts,color=pg.glColor('b'),width=3.0)
            self.window.addItem(daxis)
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            # reset mesh using rotated and translated points
            self.body.setMeshData(vertexes=mesh, vertexColors=self.meshColors)

        # update the center of the camera view to the spacecraft location
        view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        self.window.opts['center'] = view_location
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

    def _get_spacecraft_points(self):
        """"
            Points that define the spacecraft, and the colors of the triangular mesh
            Define the points on the aircraft following diagram in Figure C.3
        """
        #points are in NED coordinates
        points = np.genfromtxt ('polyvert3.csv', delimiter=",")
        #print(points.shape[0])
        points = points.T
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points = R @ points
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        points = R @ points


        # scale points for better rendering
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
        mesh2 = np.genfromtxt ('polyface3.csv', delimiter=",")
        mesh3 = np.array(list(map(lambda x: list(map(lambda y: points[int(y)], x)), mesh2)))
        return mesh3

    def _Euler2Rotation(self, phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                           [0, c_phi, s_phi],
                           [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                          [-s_psi, c_psi, 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial
