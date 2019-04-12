import numpy as np
from message_types.msg_waypoints import msg_waypoints
import parameters.planner_parameters as PLAN
from chap11.dubins_parameters import dubins_parameters
from tools.tools import mod

# debugging
#import matplotlib.pyplot as plt
#import time


class Tree():
    def __init__(self,N,E,D,parent=None,goal=False,cost=0.0,chi=0.0):
        self.NED = np.array([N,E,D])
        self.parent = parent
        self.goal = goal
        self.cost = cost
        self.chi = chi


class planRRT():
    def __init__(self, map,world_view):
        self.segmentLength = 50.0 # standard length of path segments
        self.pd = 0.0
        self.map = map
        self.map_size = PLAN.city_width
        self.world_view = world_view
        self.pts = []
        self.num_paths = 5
        self.path_options = []
        self.cost_options = []


        north = self.map.building_north.reshape((PLAN.num_blocks**2,1))
        east = self.map.building_east.reshape((PLAN.num_blocks**2,1))
        self.obstacles = np.hstack((east,north))
        self.clearance = 2.0*map.building_width


    def planStraightPath(self, wpp_start, wpp_end, map, wp_msg):
        """
        Straight line dubins path planner
        """
        # desired down position is down position of end node
        self.pd = wpp_end.item(2)
        self.end = wpp_end

        # create Tree class of start node
        start_node = Tree(wpp_start.item(0), wpp_start.item(1), self.pd)



        while len(self.path_options) < self.num_paths:

            # initialize paths each run through
            self.node_iteration = 0
            # establish tree starting with the start node
            self.tree = [start_node]
            self.pts = []


            while self.tree[-1].goal == False:
                new_info = self.generateRandomNode()
                if new_info != None:
                    new_node = Tree(new_info[0],new_info[1],new_info[2],new_info[3],new_info[4],new_info[5])
                    self.tree.append(new_node)
                    self.pts.append(np.array([new_node.NED.tolist(),
                                              self.tree[new_node.parent].NED.tolist()]))
                    self.world_view.updateRRT(len(self.path_options),np.array(self.pts))

            path = []
            cost = 0.0
            # last index of tree
            self.index = len(self.tree) - 1

            # keeps appending to path while there's a parent
            while self.index != None:
                path.append(self.tree[self.index].NED)
                self.index = self.tree[self.index].parent

            # smooth the path
            path,cost = self.smoothPath(path)

            # append path and cost to path/cost options
            self.path_options.append(path)
            self.cost_options.append(cost)

        # choose the best path options
        lowest_index = np.argmin(np.array(self.cost_options))
        path = self.path_options[lowest_index]

        wp_msg.num_waypoints = len(path)
        for ii in range(len(path)):
            # reverses directiona and adds wp_msg
            wp_msg.ned[:,ii] = np.array([path[-ii-1][1],path[-ii-1][0],path[-ii-1][2]]).T

        return wp_msg

    def generateRandomNode(self):
        # genereate random configuration
        self.node_iteration += 1
        # routinely checks the end point
        if self.node_iteration % 20 == 0:
            rand_N = self.end.item(0)
            rand_E = self.end.item(1)
        else:
            rand_N = np.random.uniform(-self.map_size/2.,self.map_size/2.)
            rand_E = np.random.uniform(-self.map_size/2.,self.map_size/2.)

        # calculate closest point
        closest = np.inf*np.ones((2,1))
        for ii in range(len(self.tree)):
            distance = np.linalg.norm(self.tree[ii].NED[0:2]-np.array([rand_N,rand_E]))
            if distance < closest[1]:
                closest[0] = ii
                closest[1] = distance
        parent = int(closest[0])

        chi = np.arctan2(rand_E-self.tree[parent].NED[1],rand_N-self.tree[parent].NED[0])
        N = self.segmentLength*np.cos(chi)+self.tree[parent].NED[0]
        E = self.segmentLength*np.sin(chi)+self.tree[parent].NED[1]
        D = self.pd

        # check for collisions
        if self.checkValidity(np.array([N,E,D]),self.tree[parent].NED):

            # check to see if it reached the goal
            if np.linalg.norm(np.array([N,E,D])-self.end) < 2.*self.segmentLength:

                N = self.end[0]
                E = self.end[1]
                D = self.end[2]
                goal = True
                # calculate cost of line
                cost = np.linalg.norm(self.end-self.tree[parent].NED)
            else:
                goal = False
                cost = self.segmentLength

            # calculate cost of line


            return N,E,D,parent,goal,cost

        else:
            # return None if there is collision
            return None

    def checkValidity(self,pt1,pt2):
        percent = np.linspace(0.0,1.0,int(self.segmentLength))
        points = (np.outer(percent,pt1) + np.outer((1.0-percent),pt2))
        for ii in range(points.shape[0]):
            for jj in range(self.obstacles.shape[0]):
                if np.linalg.norm(self.obstacles[jj,:]-points[ii,0:2]) < self.clearance:
                    return False
        return True

    def smoothPath(self,path):
        new_cost = 0.0
        cur_node = 0
        next_node = 1

        new_path = [path[cur_node]]

        for ii in range(len(path)-1):
            if self.checkValidity(path[cur_node],path[next_node]):
                next_node += 1
            else:
                new_cost += np.linalg.norm(path[next_node]-path[cur_node])
                cur_node = next_node
                next_node += 1
                new_path.append(path[cur_node])


        new_path.append(path[len(path)-1])

        return new_path, new_cost



    """













    Above this break are the straight line RRT functions
    Below this break are the duibins RRT functions











    """

    def planDubinsPath(self, wpp_start, wpp_end, map, wp_msg, min_radius):
        """
        Dubins dubins path planner
        """
        self.dubins_path = dubins_parameters()
        # desired down position is down position of end node
        self.pd = wpp_end.item(2)
        self.end = wpp_end
        self.segmentLength = 3.0*min_radius

        # create Tree class of start node
        start_node = Tree(wpp_start.item(0), wpp_start.item(1), self.pd)


        while len(self.path_options) < self.num_paths:

            # initialize paths each run through
            self.node_iteration = 0
            # establish tree starting with the start node
            self.tree = [start_node]
            self.pts = []


            while self.tree[-1].goal == False:
                new_info = self.generateRandomDubinsNode(min_radius)
                if new_info != None:
                    new_node = Tree(new_info[0],new_info[1],new_info[2],new_info[3],new_info[4],new_info[5],new_info[6])
                    self.tree.append(new_node)

            path_objects = []
            cost = 0.0
            # last index of tree
            self.index = len(self.tree) - 1

            # keeps appending to path while there's a parent
            while self.index != None:
                path_objects.append(self.tree[self.index])
                self.index = self.tree[self.index].parent

            # smooth the path
            path,cost = self.smoothDubinsPath(path_objects)

            # append path and cost to path/cost options
            self.path_options.append(path)
            self.cost_options.append(cost)

        # choose the best path options
        lowest_index = np.argmin(np.array(self.cost_options))
        path = self.path_options[lowest_index]

        wp_msg.num_waypoints = len(path)
        for ii in range(len(path)):
            # reverses directiona and adds wp_msg
            wp_msg.ned[:,ii] = np.array([path[ii][1],path[ii][0],path[ii][2]]).T

        # if dubins, calculate course
        if wp_msg.type == 'dubins':
            for ii in range(len(path)-1):
                wp_msg.course[0,ii] = np.arctan2(wp_msg.ned[1,ii+1]-wp_msg.ned[1,ii],wp_msg.ned[0,ii+1]-wp_msg.ned[0,ii])

        return wp_msg



    def generateRandomDubinsNode(self,min_radius):
        # genereate random configuration
        self.radius = min_radius
        self.node_iteration += 1
        # routinely checks the end point
        if self.node_iteration % 20 == 0:
            rand_N = self.end.item(0)
            rand_E = self.end.item(1)
        else:
            rand_N = np.random.uniform(-self.map_size/2.,self.map_size/2.)
            rand_E = np.random.uniform(-self.map_size/2.,self.map_size/2.)

        # calculate closest point
        closest = np.inf*np.ones((2,1))
        for ii in range(len(self.tree)):
            distance = np.linalg.norm(self.tree[ii].NED[0:2]-np.array([rand_N,rand_E]))
            if distance < closest[1]:
                closest[0] = ii
                closest[1] = distance
        parent = int(closest[0])

        chi = np.arctan2(rand_E-self.tree[parent].NED[1],rand_N-self.tree[parent].NED[0])
        N = self.segmentLength*np.cos(chi)+self.tree[parent].NED[0]
        E = self.segmentLength*np.sin(chi)+self.tree[parent].NED[1]
        D = self.pd

        ps = np.array([self.tree[parent].NED]).T
        chis = self.tree[parent].chi
        pe = np.array([np.array([N,E,D])]).T
        chie = chi
        R = min_radius
        #self.path.orbit_radius = R
        self.dubins_path.update(ps, chis, pe, chie, R)

        # check for collisions
        valid,all_the_pts = self.checkDubinsValidity()

        if valid:

            # check to see if it reached the goal
            if np.linalg.norm(np.array([N,E,D])-self.end) < 2.*self.segmentLength:

                N = self.end[0]
                E = self.end[1]
                D = self.end[2]
                goal = True
                # calculate cost of line
                cost = self.dubins_path.length
            else:
                goal = False
                # calculate cost of line
                cost = self.dubins_path.length


            # update plots
            for ii in range(1,all_the_pts.shape[0]):
                self.pts.append(np.array([all_the_pts[ii-1].tolist(),
                                all_the_pts[ii].tolist()]))
            self.world_view.updateRRT(len(self.path_options),np.array(self.pts))

            return N,E,D,parent,goal,cost,chi

        else:
            # return None if there is collision
            return None

    def checkDubinsValidity(self):
        points = self.dubins_points(0.1)
        for ii in range(points.shape[0]):
            for jj in range(self.obstacles.shape[0]):
                if np.linalg.norm(self.obstacles[jj,:]-points[ii,0:2]) < self.clearance:
                    return False, None
        return True, points


    def dubins_points(self,Del):
        initialize_points = True
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

        #R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])
        #points = points @ R.T
        return points

    def smoothDubinsPath(self,path_objects):
        new_cost = 0.0
        cur_node = 0
        next_node = 1

        path_objects = path_objects[::-1]

        new_path = [path_objects[cur_node].NED]

        for ii in range(len(path_objects)-1):
            ps = np.array([path_objects[cur_node].NED]).T
            chis = path_objects[cur_node].chi
            pe = np.array([path_objects[next_node].NED]).T
            chie = path_objects[next_node].chi
            R = self.radius
            self.dubins_path.update(ps, chis, pe, chie, R)

            valid,points = self.checkDubinsValidity()
            if valid:
                next_node += 1
            else:
                new_cost += self.dubins_path.length
                cur_node = next_node
                next_node += 1
                new_path.append(path_objects[cur_node].NED)

        # calculate length for last node
        ps = np.array([path_objects[cur_node].NED]).T
        chis = path_objects[cur_node].chi
        pe = np.array([path_objects[-1].NED]).T
        chie = path_objects[-1].chi
        R = self.radius
        self.dubins_path.update(ps, chis, pe, chie, R)
        new_cost += self.dubins_path.length

        new_path.append(path_objects[-1].NED)

        return new_path, new_cost
