import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot
from tools.tools import RotationMatrix, wrapAnglePi2Pi
import chap5.trim_results as TR
import parameters.aerosonde_parameters as P

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(45.0)  # approach angle for large distance from straight-line path
        self.k_path = 0.02  # proportional gain for straight-line path following
        self.k_orbit = 5.0  # proportional gain for orbit following
        self.gravity = P.gravity
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot
        self.Va = TR.Va

    def update(self, path, state):
        if path.type=='line':
            self._follow_straight_line(path, state)
        elif path.type=='orbit':
            self._follow_orbit(path, state)
        return self.autopilot_commands

    def _follow_straight_line(self, path, state):
        # trim result of airpseed
        self.autopilot_commands.airspeed_command = self.Va

        # course command
        chi_q = atan2(path.line_direction.item(1),path.line_direction.item(0))
        chi_q = wrapAnglePi2Pi(chi_q)
        Ri2P = RotationMatrix(chi_q)
        p = np.array([[state.pn, state.pe, -state.h]]).T
        ep = Ri2P @ (p - path.line_origin)
        chi_d = chi_q -self.chi_inf*(2./np.pi)*atan(self.k_path*ep.item(1))
        self.autopilot_commands.course_command = chi_d

        # altitude command
        q = path.line_direction[:,0]
        ki = np.array([0.,0.,1.]).T
        n = np.cross(q,ki)/np.linalg.norm(np.cross(q,ki))
        n.resize(3,1)
        si = ep - (ep.T @ n)*n
        rd = path.line_origin.item(2)
        hd = -rd + np.sqrt(si.item(0)**2+si.item(1)**2)*(q.item(2)/np.sqrt(q.item(0)**2+q.item(1)**2))
        self.autopilot_commands.altitude_command = hd

        # feed forward same as chi desired
        self.autopilot_commands.phi_feedforward = 0.

    def _follow_orbit(self, path, state):
        self.autopilot_commands.airspeed_command = self.Va
        self.autopilot_commands.altitude_command = -path.orbit_center.item(2)

        # course angle command
        d = np.sqrt((state.pn-path.orbit_center.item(0))**2+(state.pe-path.orbit_center.item(1))**2)
        phi = atan2((state.pe-path.orbit_center.item(1)),(state.pn-path.orbit_center.item(0)))
        phi = wrapAnglePi2Pi(phi)
        if path.orbit_direction == 'CW':
            lamb = 1.
        else:
            lamb = -1.
        rho = path.orbit_radius
        position = np.array([[state.pn,state.pe,-state.h]]).T
        d = np.linalg.norm(position-path.orbit_center)
        chi_c = phi + lamb*(np.pi/2. + atan(self.k_orbit*((d-rho)/rho)))
        self.autopilot_commands.course_command = chi_c
        wind_down = 0.
        part_a = state.Va**2
        part_b = (state.wn*sin(state.chi)-state.we*cos(state.chi))**2+(wind_down)**2
        if part_a >= part_b:
            stuff = part_a - part_b
        else:
            stuff = 0.0
        vg_sqrd = (state.wn*cos(state.chi)+state.we*sin(state.chi)+np.sqrt(stuff))**2
        if state.Va == 0:
            state.Va = self.Va
        denom = self.gravity*rho*np.sqrt((state.Va**2-(state.wn*sin(state.chi)-state.we*cos(state.chi))**2-(wind_down)**2)/(state.Va**2-wind_down**2))
        phi_ff = lamb*atan2(vg_sqrd,denom)
        phi_ff = wrapAnglePi2Pi(phi_ff)
        self.autopilot_commands.phi_feedforward = phi_ff
