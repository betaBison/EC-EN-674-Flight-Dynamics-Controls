import numpy as np
from message_types.msg_waypoints import msg_waypoints
import parameters.planner_parameters as PLAN

# debugging
import matplotlib.pyplot as plt
import time


class Tree():
    def __init__(self,N,E,D,parent=None,goal=False,cost=0.0,):
        self.NED = np.array([N,E,D])
        self.parent = parent
        self.goal = goal
        self.cost = cost


class planRRT():
    def __init__(self, map,world_view):
        self.waypoints = msg_waypoints()
        self.segmentLength = 100.0 # standard length of path segments
        self.pd = 0.0
        self.map = map
        self.map_size = PLAN.city_width
        self.world_view = world_view
        self.pts = []

        north = self.map.building_north.reshape((PLAN.num_blocks**2,1))
        east = self.map.building_east.reshape((PLAN.num_blocks**2,1))
        self.obstacles = np.hstack((east,north))
        self.clearance = map.building_width


    def planPath(self, wpp_start, wpp_end, map):

        # desired down position is down position of end node
        self.pd = wpp_end.item(2)
        self.end = wpp_end

        #print("start=",wpp_start)

        # specify start and end nodes from wpp_start and wpp_end
        # format: N, E, D, cost, parentIndex, connectsToGoalFlag,
        #start_node = np.array([wpp_start.item(0), wpp_start.item(1), pd, 0, 0, 0])
        #end_node = np.array([wpp_end.item(0), wpp_end.item(1), pd, 0, 0, 0])

        start_node = Tree(wpp_start.item(0), wpp_start.item(1), self.pd)
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
                self.world_view.updateRRT(np.array(self.pts))

                #for ii in range(1,len(self.tree)):
                #    plt.plot([self.tree[self.tree[ii].parent].NED[0],self.tree[ii].NED[0]],[self.tree[self.tree[ii].parent].NED[1],self.tree[ii].NED[1]],'g')
                #plt.show()

        return None
        """
        # check to see if start_node connects directly to end_node
        if ((np.linalg.norm(start_node[0:3] - end_node[0:3]) < self.segmentLength ) and not self.collision(start_node, end_node, map)):
            self.waypoints.ned = end_node[0:3]
        else:
            numPaths = 0
            while numPaths < 3:
                tree, flag = self.extendTree(tree, end_node, self.segmentLength, map, pd)
                numPaths = numPaths + flag
        """


        # find path with minimum cost to end_node
        #path = self.findMinimumPath(tree, end_node)
        #return self.smoothPath(path, map)

    def generateRandomNode(self):
        # genereate random configuration
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
            if np.linalg.norm(np.array([N,E,D])-self.end) < 100.0:
                N = self.end[0]
                E = self.end[1]
                D = self.end[2]
                goal = True
            else:
                goal = False

            # calculate cost of line
            cost = np.linalg.norm(self.end-np.array([N,E,D]))

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

    def collision(start_node, end_node, map):
        pass

    def pointsAlongPath(start_node, end_node, Del):
        pass
    def downAtNE(map, n, e):
        pass
    def extendTree(tree, end_node, segmentLength, map, pd):
        pass
    def findMinimumPath(tree, end_node):
        pass
    def smoothPath(path, map):
        pass
