import sys
sys.path.append('..')
from message_types.msg_waypoints import msg_waypoints
import parameters.planner_parameters as PLAN
import parameters.aerosonde_parameters as P

from scipy.spatial import Voronoi
import numpy as np
import math
import time
import voronoiTools as VT

# debugging
import matplotlib.pyplot as plt
from scipy.spatial import voronoi_plot_2d

class Voronoi_Planner():
    def __init__(self,map):
        self.map = map
        self.num_closest_points = 3
        self.end = np.array([[PLAN.city_width/2.,PLAN.city_width/2.]])
        start = np.array([[P.pn0, P.pe0,]])
        start = np.array([np.transpose([start[0,1],start[0,0]])])




        north = self.map.building_north.reshape((PLAN.num_blocks**2,1))
        east = self.map.building_east.reshape((PLAN.num_blocks**2,1))
        self.points = np.hstack((east,north))


        self.E = [] # edges array
        vor = Voronoi(self.points)
        self.ridge = vor.ridge_vertices
        self.V = np.concatenate((vor.vertices,start,self.end),axis=0)
        num_vertices = len(vor.vertices)
        self.ridge_points = vor.ridge_points

        # debugging
        #voronoi_plot_2d(vor,line_colors='red',show_vertices=True)
        #plt.show()


        for vpair in self.ridge:
            if vpair[0] >= 0 and vpair[1] >= 0:
                v0 = vor.vertices[vpair[0]]
                v1 = vor.vertices[vpair[1]]
                self.E.append([[v0[0], v0[1]], [v1[0], v1[1]]])


        if num_vertices < self.num_closest_points:
            self.num_closest_points = num_vertices
        closest = np.ones((self.num_closest_points,2),dtype=float)
        closest*=PLAN.city_width*1e90
        for ii in range(num_vertices):
            distance = VT.calcDistance(vor.vertices[ii],start[0])
            if distance < np.amax(closest[:,0]):
                closest[np.argmax(closest[:,0]),1] = ii
                closest[np.argmax(closest[:,0]),0] = distance
        start_index = num_vertices
        for ii in range(self.num_closest_points):
            closest_index = int(closest[ii,1])
            self.ridge.append([closest_index,start_index])
            self.E.append([[vor.vertices[int(closest[ii,1]),0], vor.vertices[int(closest[ii,1]),1]],[start[0][0],start[0][1]]])
        closest = np.ones((self.num_closest_points,2),dtype=float)
        closest*=PLAN.city_width*1e9
        for ii in range(num_vertices):
            distance = VT.calcDistance(vor.vertices[ii],self.end[0])
            if distance < np.amax(closest[:,0]):
                closest[np.argmax(closest[:,0]),1] = ii
                closest[np.argmax(closest[:,0]),0] = distance
        end_index = num_vertices+1
        for ii in range(self.num_closest_points):
            closest_index = int(closest[ii,1])
            self.ridge.append([closest_index,end_index])
            self.E.append([[vor.vertices[int(closest[ii,1]),0], vor.vertices[int(closest[ii,1]),1]],[self.end[0][0],self.end[0][1]]])
        self.ridge.append([start_index,end_index])
        #self.E.append([[start[0][0],start[0][1]],[self.end[0][0],self.end[0][1]]])
        self.E.append([[start[0][0],start[0][1]],[self.end[0][0],self.end[0][1]]])
        self.E = np.asarray(self.E)


        self.E_inf = []
        center = self.points.mean(axis=0)
        for pointidx, simplex in zip(self.ridge_points, self.ridge):
            simplex = np.asarray(simplex)
            if np.any(simplex < 0):
                i = simplex[simplex >= 0][0] # finite end Voronoi vertex
                t = self.points[pointidx[1]] - self.points[pointidx[0]]  # tangent
                t = t / np.linalg.norm(t)
                n = np.array([-t[1], t[0]]) # normal
                midpoint = self.points[pointidx].mean(axis=0)
                far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * PLAN.city_width*10.0
                self.E_inf.append([[vor.vertices[i,0], vor.vertices[i,1]],[far_point[0], far_point[1]]])
                #plt.plot([vor.vertices[i,0], far_point[0]],[vor.vertices[i,1], far_point[1]], 'g--')
        self.E_inf = np.asarray(self.E_inf)

        # assign weight
        weight = np.zeros((self.E.shape[0],1))
        self.ridge = list(filter(lambda x: x[0] >= 0, self.ridge))
        for ii in range(len(self.ridge)):
            D_prime = np.zeros((self.points.shape[0],1))
            for jj in range(self.points.shape[0]):
                sigma_star = VT.calcSigmaStar(self.points[jj],self.E[ii,0,:],self.E[ii,1,:])
                if sigma_star < 0.0:
                    D_prime[jj] = np.linalg.norm(self.points[jj]-self.E[ii,0,:])
                elif sigma_star > 1.0:
                    D_prime[jj] = np.linalg.norm(self.points[jj]-self.E[ii,1,:])
                else:
                    D_prime[jj] = VT.calcWeight(self.points[jj],self.E[ii,0,:],self.E[ii,1,:])
            weight[ii] = VT.calcCost(self.E[ii,0,:],self.E[ii,1,:],np.amin(D_prime))
            if np.amin(D_prime) < 0:
                print(np.amin(D_prime))

        path = VT.dijkstraSearch(self.V,self.ridge,weight)

        '''
        path_weight = np.zeros((len(path),1))
        for mm in range(len(path)):
            path_weight[mm] = weight[mm]
        print(path_weight)
        '''

        self.path_pts = []
        for ii in range(len(path)-1):
            for jj in range(len(self.ridge)):
                if path[ii] < path[ii+1]:
                    a = path[ii]
                    b = path[ii+1]
                else:
                    a = path[ii+1]
                    b = path[ii]
                if [a,b] == self.ridge[jj]:
                    self.path_pts.append(self.E[jj,:,:])
                    break
        self.path_pts = np.asarray(self.path_pts)




        #return self.V[path[1]],start,np.amax(weight)


        # waypoints return message
        self.waypoints = msg_waypoints()
        self.waypoints.type = 'fillet'
        self.waypoints.num_waypoints = len(path)
        Va = PLAN.Va0
        self.waypoints.airspeed[:, 0:self.waypoints.num_waypoints] \
            = Va*np.ones((1,self.waypoints.num_waypoints))

        path_plan = np.zeros((3,len(path)))
        for ii in range(len(path)):
            path_plan[:,ii] = [self.V[path[ii]].item(1),self.V[path[ii]].item(0),-100.]
        self.waypoints.ned[:, 0:self.waypoints.num_waypoints] \
            = path_plan

        course_plan = np.zeros((1,len(path)))
        for ii in range(0,len(path)-1):
            #print(self.waypoints.ned[1,ii+1]-self.waypoints.ned[1,ii])
            #print(np.arctan2(self.waypoints.ned[1,ii+1]-self.waypoints.ned[1,ii],self.waypoints.ned[0,ii+1]-self.waypoints.ned[0,ii]))
            course_plan[0,ii] = np.arctan2(self.waypoints.ned[1,ii+1]-self.waypoints.ned[1,ii],self.waypoints.ned[0,ii+1]-self.waypoints.ned[0,ii])
        self.waypoints.course[:, 0:self.waypoints.num_waypoints] \
            = course_plan


    def plan(self):
        return self.waypoints
