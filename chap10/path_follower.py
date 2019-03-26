import numpy as np
from math import sin, cos, atan, atan2
import sys

sys.path.append('..')
from message_types.msg_autopilot import msg_autopilot
from tools.tools import RotationMatrix, wrapAnglePi2Pi
import chap5.trim_results as TR

class path_follower:
    def __init__(self):
        self.chi_inf = np.radians(45.0)  # approach angle for large distance from straight-line path
        self.k_path = 0.02  # proportional gain for straight-line path following
        self.k_orbit = 1.0  # proportional gain for orbit following
        self.gravity = 9.8
        self.autopilot_commands = msg_autopilot()  # message sent to autopilot
        self.Va = TR.Va

    def update(self, path, state):
        if path.flag=='line':
            self._follow_straight_line(path, state)
        elif path.flag=='orbit':
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
        self.autopilot_commands.phi_feedforward = chi_d

    def _follow_orbit(self, path, state):
        self.autopilot_commands.airspeed_command = 0.0
        self.autopilot_commands.course_command = 0.0
        self.autopilot_commands.altitude_command = 0.0
        self.autopilot_commands.phi_feedforward = 0.0
