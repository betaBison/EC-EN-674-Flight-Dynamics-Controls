import numpy as np
import sys
sys.path.append('..')
from chap11.dubins_parameters import dubins_parameters
from parameters import planner_parameters as PP
from message_types.msg_path import msg_path


class path_manager:
    def __init__(self):
        # message sent to path follower
        self.path = msg_path()
        # pointers to previous, current, and next waypoints
        self.ptr_previous = 0
        self.ptr_current = 1
        self.ptr_next = 2
        # flag that request new waypoints from path planner
        self.flag_need_new_waypoints = True
        self.num_waypoints = 0
        self.halfspace_n = np.inf * np.ones((3,1))
        self.halfspace_r = np.inf * np.ones((3,1))
        # state of the manager state machine
        self.manager_state = 1
        self.attempt = True
        self.recalculate = False
        # dubins path parameters
        self.dubins_path = dubins_parameters()
        self.R = PP.R_min

    def update(self, waypoints, radius, state):
        # this flag is set for one time step to signal a redraw in the viewer
        if self.path.flag_path_changed == True:
            self.path.flag_path_changed = False
        if waypoints.num_waypoints == 0:
            waypoints.flag_manager_requests_waypoints = True
            self.num_waypoints = waypoints.num_waypoints
        else:
            if waypoints.type == 'straight_line':
                self.line_manager(waypoints, state)
            elif waypoints.type == 'fillet':
                self.fillet_manager(waypoints, radius, state)
            elif waypoints.type == 'dubins':
                self.dubins_manager(waypoints, radius, state)
            else:
                print('Error in Path Manager: Undefined waypoint type.')
        return self.path

    def line_manager(self, waypoints, state):
        wi_prev = np.array(waypoints.ned[:,self.ptr_previous])
        wi_cur = np.array(waypoints.ned[:,self.ptr_current])
        wi_next = np.array(waypoints.ned[:,self.ptr_next])
        self.halfspace_r = np.array([wi_cur]).T


        qi_prev = (wi_cur - wi_prev)/np.linalg.norm(wi_cur - wi_prev)
        qi_cur = (wi_next - wi_cur)/np.linalg.norm(wi_next - wi_cur)
        ni = (qi_prev - qi_cur)/np.linalg.norm(qi_prev - qi_cur)
        self.halfspace_n = np.array([ni]).T

        # go to the next path if you're in the half space
        if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
            self.increment_pointers()

        self.path.type = 'line'
        self.path.line_origin = np.array([waypoints.ned[:,self.ptr_previous]]).T
        diff = waypoints.ned[:,self.ptr_current] - waypoints.ned[:,self.ptr_previous]
        self.path.line_direction = np.array([diff/np.linalg.norm(diff)]).T


    def fillet_manager(self, waypoints, radius, state):
        wi_prev = np.array([waypoints.ned[:,self.ptr_previous]]).T
        wi_cur = np.array([waypoints.ned[:,self.ptr_current]]).T
        wi_next = np.array([waypoints.ned[:,self.ptr_next]]).T

        qi_prev = (wi_cur - wi_prev)/np.linalg.norm(wi_cur - wi_prev)
        qi_cur = (wi_next - wi_cur)/np.linalg.norm(wi_next - wi_cur)
        rose = np.arccos(-qi_prev.T @ qi_cur)
        if self.manager_state == 1:
            self.path.line_origin = np.array([waypoints.ned[:,self.ptr_previous]]).T
            diff = waypoints.ned[:,self.ptr_current] - waypoints.ned[:,self.ptr_previous]
            self.path.line_direction = np.array([diff/np.linalg.norm(diff)]).T
            self.halfspace_r = wi_cur - (self.R/np.tan(rose/2.))*qi_prev
            self.halfspace_n = qi_prev
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                #print("change to orbit")
                self.manager_state = 2
                self.path.flag_path_changed = True
            self.halfspace_r = wi_cur + (self.R/np.tan(rose/2.))*qi_cur
            self.halfspace_n = qi_cur
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.attempt = False
            else:
                self.attempt = True
        else: # self.manager_state == 1:
            ni = (qi_prev - qi_cur)/np.linalg.norm(qi_prev - qi_cur)
            self.path.orbit_center = wi_cur - (self.R/np.sin(rose/2.))*ni
            self.path.orbit_radius = self.R
            lamb = np.sign(qi_prev.item(0)*qi_cur.item(1)-qi_prev.item(1)*qi_cur.item(0))
            self.path.orbit_direction = self.convertDirection(lamb)
            self.halfspace_r = wi_cur + (self.R/np.tan(rose/2.))*qi_cur
            self.halfspace_n = qi_cur
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                if self.attempt:
                    #print("change to line")

                    self.recalculate = True
                    self.increment_pointers()
            else:
                self.attempt = True

        if self.recalculate:
            wi_prev = np.array(waypoints.ned[:,self.ptr_previous])
            wi_cur = np.array(waypoints.ned[:,self.ptr_current])
            wi_next = np.array(waypoints.ned[:,self.ptr_next])

            qi_prev = (wi_cur - wi_prev)/np.linalg.norm(wi_cur - wi_prev)
            qi_cur = (wi_next - wi_cur)/np.linalg.norm(wi_next - wi_cur)
            rose = np.arccos(-qi_prev.T @ qi_cur)
            self.recalculate = False

        if self.manager_state == 1:
            self.path.type = 'line'
            self.path.line_origin = np.array([waypoints.ned[:,self.ptr_previous]]).T
            diff = waypoints.ned[:,self.ptr_current] - waypoints.ned[:,self.ptr_previous]
            self.path.line_direction = np.array([diff/np.linalg.norm(diff)]).T
        else:
            self.path.type = 'orbit'
            ni = (qi_prev - qi_cur)/np.linalg.norm(qi_prev - qi_cur)
            self.path.orbit_center = wi_cur - (self.R/np.sin(rose/2.))*ni
            self.path.orbit_radius = self.R
            lamb = np.sign(qi_prev.item(0)*qi_cur.item(1)-qi_prev.item(1)*qi_cur.item(0))
            self.path.orbit_direction = self.convertDirection(lamb)

    def dubins_manager(self, waypoints, radius, state):
        ps = np.array([waypoints.ned[:,self.ptr_previous]]).T
        chis = waypoints.course.item(self.ptr_previous)
        pe = np.array([waypoints.ned[:,self.ptr_current]]).T
        chie = waypoints.course.item(self.ptr_current)
        R = radius
        #self.path.orbit_radius = R
        self.dubins_path.update(ps, chis, pe, chie, R)
        #print(self.manager_state)
        if self.manager_state == 1:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_s
            self.path.orbit_radius = self.dubins_path.radius
            self.path.orbit_direction = self.convertDirection(self.dubins_path.dir_s)

            # half space variables
            self.halfspace_r = self.dubins_path.r1
            self.halfspace_n = -self.dubins_path.n1
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.path.flag_path_changed = True
                self.manager_state = 2
        if self.manager_state == 2:
            # half space variables
            self.halfspace_r = self.dubins_path.r1
            self.halfspace_n = self.dubins_path.n1
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.path.flag_path_changed = True
                self.manager_state = 3
        if self.manager_state == 3:
            self.path.type = 'line'
            self.path.line_origin = self.dubins_path.r1
            self.path.line_direction = self.dubins_path.n1/np.linalg.norm(self.dubins_path.n1)
            # half space variables
            self.halfspace_r = self.dubins_path.r2
            self.halfspace_n = self.dubins_path.n1
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.path.flag_path_changed = True
                self.manager_state = 4
        if self.manager_state == 4:
            self.path.type = 'orbit'
            self.path.orbit_center = self.dubins_path.center_e
            self.path.orbit_radius = self.dubins_path.radius

            self.path.orbit_direction = self.convertDirection(self.dubins_path.dir_e)

            # half space variables
            self.halfspace_r = self.dubins_path.r3
            self.halfspace_n = -self.dubins_path.n3
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.path.flag_path_changed = True
                self.manager_state = 5
        if self.manager_state == 5:
            #print("direction = ",self.dubins_path.dir_e)
            # half space variables
            self.halfspace_r = self.dubins_path.r3
            self.halfspace_n = self.dubins_path.n3
            if self.inHalfSpace(np.array([[state.pn,state.pe,-state.h]]).T):
                self.increment_pointers()



    def initialize_pointers(self):
        pass

    def increment_pointers(self):
        self.manager_state = 1
        self.path.flag_path_changed = True
        self.ptr_previous += 1
        # wrap to start if reach the end
        if self.ptr_previous == self.num_waypoints:
            self.ptr_previous -= self.num_waypoints
        self.ptr_current += 1
        # wrap to start if reach the end
        if self.ptr_current == self.num_waypoints:
            self.ptr_current -= self.num_waypoints
        self.ptr_next += 1
        # wrap to start if reach the end
        if self.ptr_next == self.num_waypoints:
            self.ptr_next -= self.num_waypoints

    def inHalfSpace(self, pos):
        if (pos-self.halfspace_r).T @ self.halfspace_n >= 0:
            return True
        else:
            return False

    def convertDirection(self,lamb):
        if lamb == 1:
            result = 'CW'
        else:
            result = 'CCW'
        return result
