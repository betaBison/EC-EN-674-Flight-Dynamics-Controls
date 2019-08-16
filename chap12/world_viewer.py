"""
mavsim_python: world viewer (for chapter 12)
    - Beard & McLain, PUP, 2012
    - Update history:
        4/3/2019 - BGM
"""
import sys
sys.path.append("..")
import numpy as np

import pyqtgraph as pg
import pyqtgraph.opengl as gl
import pyqtgraph.Vector as Vector
from PyQt5 import QtWidgets

from tools.tools import RotationBody2Vehicle
from chap11.dubins_parameters import dubins_parameters

class world_viewer():
    def __init__(self,map,voronoi = None):
        self.scale = 4000
        # initialize Qt gui application and window
        self.app = pg.QtGui.QApplication([])  # initialize QT
        self.window = gl.GLViewWidget()  # initialize the view object
        self.window.setWindowTitle('World Viewer')
        #self.window.setGeometry(0, 0, 1500, 1500)  # args: upper_left_x, upper_right_y, width, height
        sg = QtWidgets.QDesktopWidget().availableGeometry()
        self.window.setGeometry(sg.width()/2.,0,sg.width()/2.,sg.height())
        grid = gl.GLGridItem() # make a grid to represent the ground
        grid.scale(self.scale/20, self.scale/20, self.scale/20) # set the size of the grid (distance between each line)
        self.window.addItem(grid) # add grid to viewer
        self.window.setCameraPosition(distance=self.scale, elevation=50, azimuth=-90)
        self.window.setBackgroundColor('k')  # set background color to black
        self.window.show()  # display configured window
        self.window.raise_() # bring window to the front
        self.plot_initialized = False # has the mav been plotted yet?
        # get points that define the non-rotated, non-translated mav and the mesh colors
        self.mav_points, self.mav_meshColors = self.get_mav_points()
        # dubins path parameters
        self.dubins_path = dubins_parameters()
        self.mav_body = []

        if voronoi != None:
            self.voronoi = voronoi
            vm_all_pts = self.voronoi.E_inf
            self.vm_all = gl.GLLinePlotItem(pos=vm_all_pts,color=pg.glColor('w'),width=1.0,mode='lines')
            self.window.addItem(self.vm_all)
            vm_pts = self.voronoi.E
            self.vm = gl.GLLinePlotItem(pos=vm_pts,color=pg.glColor('y'),width=1.0,mode='lines')
            self.window.addItem(self.vm)
            #vm_path_pts = self.voronoi.path_pts
            #self.vm_path = gl.GLLinePlotItem(pos=vm_path_pts,color=pg.glColor('m'),width=4.0,mode='lines')
            #self.window.addItem(self.vm_path)

        # draw map
        self.drawMap(map)
        self.initialized_RRT = False
        self.RRT_iteration = 0
        self.RRT_colors = [pg.glColor('y'),pg.glColor('g'),pg.glColor('b'),pg.glColor('w'),pg.glColor('r'),pg.glColor('m')]

        #self.app.processEvents()

    ###################################
    # public functions
    def update(self, waypoints, path, state):

        # initialize the drawing the first time update() is called
        if not self.plot_initialized:
            self.drawMAV(state)
            self.drawWaypoints(waypoints, path.orbit_radius)
            self.drawPath(path)
            self.plot_initialized = True

        # else update drawing on all other calls to update()
        else:
            self.drawMAV(state)
            if waypoints.flag_waypoints_changed==True:
                self.drawWaypoints(waypoints, path.orbit_radius)
            if path.flag_path_changed==True:
                self.drawPath(path)

        # update the center of the camera view to the mav location
        #view_location = Vector(state.pe, state.pn, state.h)  # defined in ENU coordinates
        #self.window.opts['center'] = view_location
        # redraw
        self.app.processEvents()

    def updateRRT(self,iteration,rrt_pts):
        if iteration != self.RRT_iteration:
            self.initialized_RRT = False
            self.RRT_iteration = iteration
        if not self.initialized_RRT:
            """
            vm_all_pts = self.voronoi.E_inf
            self.vm_all = gl.GLLinePlotItem(pos=vm_all_pts,color=pg.glColor('w'),width=1.0,mode='lines')
            self.w.addItem(self.vm_all)
            """
            # allows repeated colors
            while iteration > len(self.RRT_colors)-1:
                iteration -= len(self.RRT_colors)
            rrt_color = self.RRT_colors[iteration]
            self.rrt_line = gl.GLLinePlotItem(pos=rrt_pts,
                                               color=rrt_color,
                                               width=1,
                                               antialias=True,
                                               mode='lines')
            self.window.addItem(self.rrt_line)
            self.initialized_RRT = True
        else:
            self.rrt_line.setData(pos=rrt_pts)

        self.app.processEvents()

    def drawMAV(self, state):
        """
        Update the drawing of the MAV.

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
        rotated_points = self.rotate_points(self.mav_points, R)
        translated_points = self.translate_points(rotated_points, mav_position)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

        translated_points = np.matmul(R, translated_points)
        # convert points to triangular mesh defined as array of three 3D points (Nx3x3)
        mesh = self.points_to_mesh(translated_points)
        if not self.plot_initialized:
            # initialize drawing of triangular mesh.
            self.mav_body = gl.GLMeshItem(vertexes=mesh,  # defines the triangular mesh (Nx3x3)
                                  vertexColors=self.mav_meshColors,  # defines mesh colors (Nx1)
                                  drawEdges=False,  # draw edges between mesh elements
                                  smooth=False,  # speeds up rendering
                                  computeNormals=False)  # speeds up rendering
            self.window.addItem(self.mav_body)  # add body to plot
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
        else:
            # draw MAV by resetting mesh using rotated and translated points
            self.mav_body.setMeshData(vertexes=mesh, vertexColors=self.mav_meshColors)

    def rotate_points(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = np.matmul(R, points)
        return rotated_points

    def translate_points(self, points, translation):
        "Translate points by the vector translation"
        translated_points = points + np.dot(translation, np.ones([1,points.shape[1]]))
        return translated_points

    def get_mav_points(self):
        #points are in NED coordinates
        points = np.genfromtxt ('../chap2/polyvert3.csv', delimiter=",")
        #print(points.shape[0])
        points = points.T
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        points = np.matmul(R, points)
        R = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        points = np.matmul(R, points)


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

    def points_to_mesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
          (a rectangle requires two triangular mesh faces)
        """
        points=points.T
        mesh2 = np.genfromtxt ('../chap2/polyface3.csv', delimiter=",")
        mesh3 = np.array(list(map(lambda x: list(map(lambda y: points[int(y)], x)), mesh2)))
        return mesh3

    def drawPath(self, path):
        if path.type == 'line':
            points = self.straight_line_points(path)
        elif path.type == 'orbit':
            points = self.orbit_points(path)
        if not self.plot_initialized:
            path_color = pg.glColor('r')
            self.path = gl.GLLinePlotItem(pos=points,
                                          color=path_color,
                                          width=2,
                                          antialias=True,
                                          mode='line_strip')
                                          #mode='line_strip')
            self.window.addItem(self.path)
        else:
            self.path.setData(pos=points)

    def straight_line_points(self, path):
        points = np.array([[path.line_origin.item(0),
                            path.line_origin.item(1),
                            path.line_origin.item(2)],
                           [path.line_origin.item(0) + self.scale * path.line_direction.item(0),
                            path.line_origin.item(1) + self.scale * path.line_direction.item(1),
                            path.line_origin.item(2) + self.scale * path.line_direction.item(2)]])
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = np.matmul(points, R.T)
        return points

    def orbit_points(self, path):
        N = 10
        theta = 0
        theta_list = [theta]
        while theta < 2*np.pi:
            theta += 0.1
            theta_list.append(theta)
        points = np.array([[path.orbit_center.item(0) + path.orbit_radius,
                            path.orbit_center.item(1),
                            path.orbit_center.item(2)]])
        for angle in theta_list:
            new_point = np.array([[path.orbit_center.item(0) + path.orbit_radius * np.cos(angle),
                                   path.orbit_center.item(1) + path.orbit_radius * np.sin(angle),
                                   path.orbit_center.item(2)]])
            points = np.concatenate((points, new_point), axis=0)
        # convert North-East Down to East-North-Up for rendering
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = np.matmul(points, R.T)
        return points

    def drawWaypoints(self, waypoints, radius):
        if waypoints.type=='straight_line' or waypoints.type=='fillet':
            points = self.straight_waypoint_points(waypoints)
        elif waypoints.type=='dubins':
            points = self.dubins_points(waypoints, radius, 0.1)
        if not self.plot_initialized:
            waypoint_color = pg.glColor('m')
            self.waypoints = gl.GLLinePlotItem(pos=points,
                                               color=waypoint_color,
                                               width=4,
                                               antialias=True,
                                               mode='line_strip')
            self.window.addItem(self.waypoints)
        else:
            self.waypoints.setData(pos=points)

    def straight_waypoint_points(self, waypoints):
        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = np.matmul(R, waypoints.ned)
        return points.T

    def dubins_points(self, waypoints, radius, Del):
        initialize_points = True
        for j in range(0, waypoints.num_waypoints-1):
            self.dubins_path.update(
                waypoints.ned[:, j:j+1],
                waypoints.course.item(j),
                waypoints.ned[:, j+1:j+2],
                waypoints.course.item(j+1),
                radius)

            # points along start circle
            th1 = np.arctan2(self.dubins_path.p_s.item(1) - self.dubins_path.center_s.item(1),
                            self.dubins_path.p_s.item(0) - self.dubins_path.center_s.item(0))
            th1 = mod(th1)
            th2 = np.arctan2(self.dubins_path.r1.item(1) - self.dubins_path.center_s.item(1),
                             self.dubins_path.r1.item(0) - self.dubins_path.center_s.item(0))
            th2 = mod(th2)
            th = th1
            theta_list = [th]
            if self.dubins_path.dir_s > 0:
                if th1 >= th2:
                    while th < th2 + 2*np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2*np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)

            if initialize_points:
                points = np.array([[self.dubins_path.center_s.item(0) + self.dubins_path.radius * np.cos(theta_list[0]),
                                    self.dubins_path.center_s.item(1) + self.dubins_path.radius * np.sin(theta_list[0]),
                                    self.dubins_path.center_s.item(2)]])
                initialize_points = False
            for angle in theta_list:
                new_point = np.array([[self.dubins_path.center_s.item(0) + self.dubins_path.radius * np.cos(angle),
                                       self.dubins_path.center_s.item(1) + self.dubins_path.radius * np.sin(angle),
                                       self.dubins_path.center_s.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

            # points along straight line
            sig = 0
            while sig <= 1:
                new_point = np.array([[(1 - sig) * self.dubins_path.r1.item(0) + sig * self.dubins_path.r2.item(0),
                                       (1 - sig) * self.dubins_path.r1.item(1) + sig * self.dubins_path.r2.item(1),
                                       (1 - sig) * self.dubins_path.r1.item(2) + sig * self.dubins_path.r2.item(2)]])
                points = np.concatenate((points, new_point), axis=0)
                sig += Del
            # points along end circle
            th2 = np.arctan2(self.dubins_path.p_e.item(1) - self.dubins_path.center_e.item(1),
                             self.dubins_path.p_e.item(0) - self.dubins_path.center_e.item(0))
            th2 = mod(th2)
            th1 = np.arctan2(self.dubins_path.r2.item(1) - self.dubins_path.center_e.item(1),
                             self.dubins_path.r2.item(0) - self.dubins_path.center_e.item(0))
            th1 = mod(th1)
            th = th1
            theta_list = [th]
            if self.dubins_path.dir_e > 0:
                if th1 >= th2:
                    while th < th2 + 2 * np.pi:
                        th += Del
                        theta_list.append(th)
                else:
                    while th < th2:
                        th += Del
                        theta_list.append(th)
            else:
                if th1 <= th2:
                    while th > th2 - 2 * np.pi:
                        th -= Del
                        theta_list.append(th)
                else:
                    while th > th2:
                        th -= Del
                        theta_list.append(th)
            for angle in theta_list:
                new_point = np.array([[self.dubins_path.center_e.item(0) + self.dubins_path.radius * np.cos(angle),
                                       self.dubins_path.center_e.item(1) + self.dubins_path.radius * np.sin(angle),
                                       self.dubins_path.center_e.item(2)]])
                points = np.concatenate((points, new_point), axis=0)

        R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        points = np.matmul(points, R.T)
        return points

    def drawMap(self, map):
        # draw map of the world: buildings
        fullMesh = np.array([], dtype=np.float32).reshape(0,3,3)
        fullMeshColors = np.array([], dtype=np.float32).reshape(0,3,4)
        for ii in range(0, map.num_city_blocks):
            for jj in range (0, map.num_city_blocks):
                mesh, meshColors = self.buildingVertFace(map.building_north[ii,jj],
                                                      map.building_east[ii,jj],
                                                      map.building_width,
                                                      map.building_height[ii, jj])
                fullMesh = np.concatenate((fullMesh, mesh), axis=0)
                fullMeshColors = np.concatenate((fullMeshColors, meshColors), axis=0)
        self.map = gl.GLMeshItem(vertexes= fullMesh,  # defines the triangular mesh (Nx3x3)
                      vertexColors= fullMeshColors,  # defines mesh colors (Nx1)
                      drawEdges=False,  # draw edges between mesh elements
                      smooth=False,  # speeds up rendering
                      computeNormals=False)  # speeds up rendering
        self.window.addItem(self.map)

    def buildingVertFace(self, n, e, width, height):
        # define patches for a building located at (x, y)
        # vertices of the building
        points = np.array([[e + width / 2, n + width / 2, 0], #NE 0
                         [e + width / 2, n - width / 2, 0],   #SE 1
                         [e - width / 2, n - width / 2, 0],   #SW 2
                         [e - width / 2, n + width / 2, 0],   #NW 3
                         [e + width / 2, n + width / 2, height], #NE Higher 4
                         [e + width / 2, n - width / 2, height], #SE Higher 5
                         [e - width / 2, n - width / 2, height], #SW Higher 6
                         [e - width / 2, n + width / 2, height]]) #NW Higher 7
        mesh = np.array([[points[0], points[3], points[4]],  #North Wall
                         [points[7], points[3], points[4]],  #North Wall
                         [points[0], points[1], points[5]],  # East Wall
                         [points[0], points[4], points[5]],  # East Wall
                         [points[1], points[2], points[6]],  # South Wall
                         [points[1], points[5], points[6]],  # South Wall
                         [points[3], points[2], points[6]],  # West Wall
                         [points[3], points[7], points[6]],  # West Wall
                         [points[4], points[7], points[5]],  # Top
                         [points[7], points[5], points[6]]])  # Top

        #   define the colors for each face of triangular mesh
        red = np.array([1., 0., 0., 1])
        green = np.array([0., 1., 0., 1])
        blue = np.array([0., 0., 1., 1])
        yellow = np.array([1., 1., 0., 1])
        grey = np.array([0.6, 0.6, 0.6, 1])
        light_grey = np.array([0.9, 0.9, 0.9, 1])
        meshColors = np.empty((10, 3, 4), dtype=np.float32)
        meshColors[0] = grey
        meshColors[1] = grey
        meshColors[2] = grey
        meshColors[3] = grey
        meshColors[4] = grey
        meshColors[5] = grey
        meshColors[6] = grey
        meshColors[7] = grey
        meshColors[8] = light_grey
        meshColors[9] = light_grey
        return mesh, meshColors


def mod(x):
    # force x to be between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x
