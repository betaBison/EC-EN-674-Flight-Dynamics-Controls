# path planner for mavsim_python
#
# mavsim_python
#     - Beard & McLain, PUP, 2012
#     - Last updated:
#         4/3/2019 - BGM
import numpy as np
import sys
sys.path.append('..')
from message_types.msg_waypoints import msg_waypoints
import parameters.planner_parameters as PLAN
import parameters.aerosonde_parameters as P
from chap12.planRRT import planRRT


class path_planner:
    def __init__(self,map,world_view):
        # waypoints definition
        self.waypoints = msg_waypoints()
        self.end = np.array([PLAN.city_width/2.0, PLAN.city_width/2.0, P.pd0])#, p.u0])
        self.start = np.array([-PLAN.city_width, -PLAN.city_width, P.pd0])#, P.u0])
        self.map = map
        self.rrt = planRRT(map,world_view)

    def update(self, map, state):
        # this flag is set for one time step to signal a redraw in the viewer
        # planner_flag = 1  # return simple waypoint path
        #planner_flag = 2  # return dubins waypoint path
        planner_flag = 3  # plan path through city using straight-line RRT
        # planner_flag = 4  # plan path through city using dubins RRT
        if planner_flag == 1:
            self.waypoints.type = 'fillet'
            self.waypoints.num_waypoints = 4
            Va = 25
            self.waypoints.ned[:, 0:self.waypoints.num_waypoints] \
                = np.array([[P.pn0, P.pe0, P.pd0],
                            [1000, 0, -100],
                            [0, 1000, -100],
                            [1000, 1000, -100]]).T
            self.waypoints.airspeed[:, 0:self.waypoints.num_waypoints] \
                = np.array([[Va, Va, Va, Va]])
        elif planner_flag == 2:
            self.waypoints.type = 'dubins'
            self.waypoints.num_waypoints = 4
            Va = PLAN.Va0
            self.waypoints.ned[:, 0:self.waypoints.num_waypoints] \
                = np.array([[P.pn0, P.pe0, P.pd0],
                            [1000., 0.0, -100.],
                            [0.0, 1000., -100.],
                            [1000., 1000., -100.]]).T
            self.waypoints.airspeed[:, 0:self.waypoints.num_waypoints] \
                = np.array([[Va, Va, Va, Va]])
            self.waypoints.course[:, 0:self.waypoints.num_waypoints] \
                = np.array([[np.radians(0),
                             np.radians(45),
                             np.radians(45),
                             np.radians(-135)]])
        elif planner_flag == 3:
            self.waypoints.type = 'fillet'
            # current configuration vector format: N, E, D, Va
            wpp_start = np.array([state.pn,
                                  state.pe,
                                  -state.h])#,
                                  #state.Va])
            if np.linalg.norm(np.array([state.pn, state.pe, -state.h])-self.end) < 10.0:
                wpp_end = self.start
            else:
                wpp_end = self.end

            waypoints = self.rrt.planPath(wpp_start, wpp_end, self.map)
            self.waypoints.ned = waypoints.ned
            self.waypoints.airspeed = waypoints.airspeed
            self.waypoints.num_waypoints = waypoints.num_waypoints
        # elif planner_flag == 4:

        else:
            print("Error in Path Planner: Undefined planner type.")

        return self.waypoints
