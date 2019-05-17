#!/usr/bin/python3.5
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tkinter import StringVar, Scale, OptionMenu, Button, Tk, HORIZONTAL, LEFT, RIGHT, X, font, Entry, Text
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tools.tools import Euler2Quaternion, Euler2Rotation, Quaternion2Euler
from stl import mesh


class EulerQuatGUI:
    def __init__(self,  window):
        self.window = window
        self.window.title('Final Project')


        scalelength = 180
        scalewidth = 18
        scalefontsize = 16
        # slider
        self.rollslider = Scale(window, from_=-90, to=90, orient=HORIZONTAL, resolution=5, label='Roll',command=self.updateRoll)
        self.rollslider.grid(column=0,row=1)
        self.rollslider.config(length=scalelength, width=scalewidth,font=("Helvetica",scalefontsize))
        self.rollslider.set(0)  # set the default position

        # slider
        self.pitchSlider = Scale(window, from_=-90, to=90, orient=HORIZONTAL, resolution=5, label='Pitch',command=self.updatePitch)
        self.pitchSlider.grid(column=0,row=2)
        self.pitchSlider.config(length=scalelength, width=scalewidth,font=("Helvetica",scalefontsize))
        self.pitchSlider.set(0)  # set the default angle

        # slider
        self.yawSlider = Scale(window, from_=-180, to=180, orient=HORIZONTAL, resolution=5, label='Yaw',command=self.updateYaw)
        self.yawSlider.grid(column=0,row=3)
        self.yawSlider.config(length=scalelength, width=scalewidth,font=("Helvetica",scalefontsize))
        self.yawSlider.set(0)  # set the default wind power

        # show me button
        self.buttonEuler = Button(window, text='Show me Euler', command=self.showMeEuler)
        self.buttonEuler.grid(column=0, row=4)
        self.buttonEuler.config(width=scalewidth,font=("Helvetica",scalefontsize))

        # show me button
        self.buttonQuat = Button(window, text='Show me Quaternion', command=self.showMeQuat)
        self.buttonQuat.grid(column=0, row=5)
        self.buttonQuat.config(width=scalewidth,font=("Helvetica",scalefontsize))



        self.e0box = Text(window, width=16, height=5)
        self.e0box.config(width=scalewidth-4,font=("Helvetica",scalefontsize))
        self.e0box.grid(column=2, row=1)

        self.ThetaBox = Text(window, width=16, height=2)
        self.ThetaBox.config(width=scalewidth-4,font=("Helvetica",scalefontsize))
        self.ThetaBox.grid(column=2, row=2)

        self.VectorBox = Text(window, width=16, height=4)
        self.VectorBox.config(width=scalewidth-4,font=("Helvetica",scalefontsize))
        self.VectorBox.grid(column=2, row=3)


        self.v0, self.v1, self.v2, self.meshColors = self.getAircraftPoints()
        self.allpoints = np.hstack((self.v0,self.v1,self.v2))

        self.initial = 0
        self.initial2 = 0
        self.phi = 0.0
        self.theta = 0.0
        self.psi = 0.0
        self.poly3d = None
        self.poly3d_num2 = None
        self.convert()


        #end of init function


    def updateRoll(self,value):
        self.phi = np.radians(float(value))
        self.convert()

    def updatePitch(self,value):
        self.theta = np.radians(float(value))
        self.convert()

    def updateYaw(self,value):
        self.psi = np.radians(float(value))
        self.convert()


    def showMeEuler(self):
        dt = [0.04]*3
        if self.psi < 0:
            dt[0] *= -1
        if self.theta < 0:
            dt[1] *= -1
        if self.phi < 0:
            dt[2] *= -1

        for i in np.arange(0.0, self.psi, dt[0]):
            self.plot2(0.0, 0.0, i)  # roll, pitch, yaw

        for i in np.arange(0.0, self.theta, dt[1]):
            self.plot2(0.0, i, self.psi)  # roll, pitch, yaw

        for i in np.arange(0.0, self.phi, dt[2]):
            self.plot2(i, self.theta, self.psi)  # roll, pitch, yaw


        self.plot2(self.phi, self.theta, self.psi)


    def showMeQuat(self):
        beginquat = np.array([[1,0,0,0]]).T
        endquat = Euler2Quaternion(self.phi, self.theta, self.psi)
        dist = np.linalg.norm(beginquat-endquat)
        howmany = int(dist*100/1.2)
        e0 = np.linspace(beginquat.item(0), endquat.item(0), num=howmany)
        e1 = np.linspace(beginquat.item(1), endquat.item(1), num=howmany)
        e2 = np.linspace(beginquat.item(2), endquat.item(2), num=howmany)
        e3 = np.linspace(beginquat.item(3), endquat.item(3), num=howmany)

        for i in range(howmany):
            quat = np.array([[ e0[i], e1[i], e2[i], e3[i] ]]).T
            phi, theta, psi = Quaternion2Euler(quat)
            self.plot2(phi, theta, psi)

        self.plot2(self.phi, self.theta, self.psi)


    def convert(self):
        quat = Euler2Quaternion(self.phi, self.theta, self.psi)
        e0 = quat.item(0)
        e1 = quat.item(1)
        e2 = quat.item(2)
        e3 = quat.item(3)

        # put the quaternion on the GUI
        self.e0box.delete(0.0, 20.20)
        self.e0box.insert(0.0, 'quaternion = \n'+str(quat))

        self.THETA  = np.arccos(e0)*2.0
        # put the number on the GUI
        self.ThetaBox.delete(0.0,20.20)
        self.ThetaBox.insert(0.0,'theta = \n'+str(round(np.degrees(self.THETA),5))+' deg')


        self.vector_v = quat[1:,:]*np.sin(self.THETA/2)
        # put the number on the GUI
        self.VectorBox.delete(0.0,20.20)
        showv = np.copy(self.vector_v)
        showv[2] = -showv[2]
        self.VectorBox.insert(0.0,'v = \n'+str(showv/np.linalg.norm(showv)))

        # make the vector_v big for plotting purposes
        test = 8*self.vector_v/np.linalg.norm(self.vector_v)
        if not np.isnan(self.vector_v.item(0)):
            R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            self.vector_v = R @ test


        self.plot()


    def rotatePoints(self, points, R):
        "Rotate points by the rotation matrix R"
        rotated_points = R @ points
        return rotated_points


    def getAircraftPoints(self):
        """"
            Define the points on the aircraft following diagram in Figure C.3
        """
        # m = mesh.Mesh.from_file('basic plane.stl')
        m = mesh.Mesh.from_file('./betterPlane.stl')
        v0 = m.v0.T
        v1 = m.v1.T
        v2 = m.v2.T
        N = v0.shape[1]
        # print('number of mesh nodes is: ',N)

        #   define the colors for each face of triangular mesh
        red = np.array([153., 0., 51., 255.])/255
        green = np.array([34., 204., 0., 255.])/255
        blue = np.array([77., 136., 255., 255.])/255
        yellow = np.array([230., 230., 0., 255.])/255
        white = np.array([255., 255., 255., 255.])/255

        meshColors = np.empty((N, 3, 4), dtype=np.float32)
        for i in range(N):
            # meshColors[i] = random.choice([red, green, blue, yellow])
            meshColors[i] = red
            if m.areas.item(i) > 0.2:
                # print(i)
                meshColors[i] = yellow


        # return points, meshColors
        return v0, v1, v2, meshColors


    def pointsToMesh(self, points):
        """"
        Converts points to triangular mesh
        Each mesh face is defined by three 3D points
        (a rectangle requires two triangular mesh faces)
        """
        N = int(points.shape[1]/3)
        translatedv0 = points[:,0:N].T
        translatedv1 = points[:,N:2*N].T
        translatedv2 = points[:,2*N:3*N].T
        my_mesh = np.zeros((N,3,3))
        for i in range(N):
            my_mesh[i] = np.array([translatedv0[i], translatedv1[i], translatedv2[i]])

        return my_mesh

    def plot (self):

        if self.initial == 0:
            fig = Figure(figsize=(9,12))
            self.ax0 = fig.add_subplot(211, projection='3d')
            self.ax1 = fig.add_subplot(212, projection='3d')
            self.canvas = FigureCanvasTkAgg(fig, master=self.window)
            self.canvas.get_tk_widget().grid(column=1,row=0,rowspan=14)
            self.ax0.set_ylim(-8,8)
            self.ax0.set_xlim(-8,8)
            self.ax0.set_zlim(-8,8)
            self.ax0.set_xlabel('j')
            self.ax0.set_ylabel('i')
            self.ax0.set_zlabel('k')

            self.ax1.set_ylim(-8,8)
            self.ax1.set_xlim(-8,8)
            self.ax1.set_zlim(-8,8)
            self.ax1.set_xlabel('j')
            self.ax1.set_ylabel('i')
            self.ax1.set_zlabel('k')


            self.ax0.grid(True)
            self.ax0.set_title('')
            self.ax1.grid(True)
            self.ax1.set_title('')

            self.initial = 1
            self.ax0.mouse_init()
            self.ax1.mouse_init()

            v = self.allpoints

            # convert North-East Down to East-North-Up for rendering
            R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            v = R @ v

            verts = self.pointsToMesh(v)

            # plot sides
            self.poly3d = Poly3DCollection(verts, alpha=.35)
            self.poly3d.set_facecolor(['blue','red'])
            self.ax0.add_collection3d(self.poly3d)

            X = [0, self.vector_v.item(0)]
            Y = [0, self.vector_v.item(1)]
            Z = [0, self.vector_v.item(2)]
            self.ELine, = self.ax0.plot(X, Y, Z, lw=4)

            self.canvas.draw()


        else:


            R = Euler2Rotation(self.phi, self.theta, self.psi)
            v = R @ self.allpoints

            # convert North-East Down to East-North-Up for rendering
            R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            v = R @ v

            verts = self.pointsToMesh(v)
            # plot sides
            self.poly3d.set_verts(verts)


            X = [0, self.vector_v.item(0)]
            Y = [0, self.vector_v.item(1)]
            Z = [0, self.vector_v.item(2)]
            self.ELine._verts3d = (X,Y,Z)

            self.canvas.draw()




    def plot2 (self, phi, theta, psi):

        if self.initial2 == 0:
            # self.ax1.cla()
            self.initial2 = 1

            v = self.allpoints
            # self.ax0.scatter3D(v[:, 0], v[:, 1], v[:, 2])
            # convert North-East Down to East-North-Up for rendering
            R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            v = R @ v
            # generate list of sides' polygons of our pyramid
            verts = self.pointsToMesh(v)

            # plot sides
            self.poly3d_num2 = Poly3DCollection(verts, alpha=.35)
            self.poly3d_num2.set_facecolor('blue')
            self.ax1.add_collection3d(self.poly3d_num2)

            self.canvas.draw()


        else:

            R = Euler2Rotation(phi, theta, psi)
            v = R @ self.allpoints

            # convert North-East Down to East-North-Up for rendering
            R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
            v = R @ v
            # self.ax0.scatter3D(v[:, 0], v[:, 1], v[:, 2])

            # generate list of sides' polygons of our pyramid
            verts = self.pointsToMesh(v)

            # plot sides
            self.poly3d_num2.set_verts(verts)

            self.canvas.draw()



window= Tk()
window.geometry("1325x1150")  #w x h
start = EulerQuatGUI(window)
window.mainloop()
