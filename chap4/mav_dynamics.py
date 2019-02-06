"""
mav_dynamics
    - this file implements the dynamic equations of motion for MAV
    - use unit quaternion for the attitude state

"""

'''
Questions:
C_ell_0 vs C_L_0 in parameter file
Using delta_t parameter for prop thrust
Final Vg, gamma, chi calculations
'''


import sys
sys.path.append('..')
import numpy as np
from math import sqrt,cos,sin,pi,atan2,asin

# load message types
from message_types.msg_state import msg_state

import parameters.aerosonde_parameters as MAV
from tools.tools import Quaternion2Euler #,Quaternion2Rotation

class mav_dynamics:
    def __init__(self, Ts):
        self._ts_simulation = Ts
        # set initial states based on parameter file
        # _state is the 13x1 internal state of the aircraft that is being propagated:
        # _state = [pn, pe, pd, u, v, w, e0, e1, e2, e3, p, q, r]
        # We will also need a variety of other elements that are functions of the _state and the wind.
        # self.true_state is a 19x1 vector that is estimated and used by the autopilot to control the aircraft:
        # true_state = [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]
        self._state = np.array([[MAV.pn0],  # (0)
                               [MAV.pe0],   # (1)
                               [MAV.pd0],   # (2)
                               [MAV.u0],    # (3)
                               [MAV.v0],    # (4)
                               [MAV.w0],    # (5)
                               [MAV.e0],    # (6)
                               [MAV.e1],    # (7)
                               [MAV.e2],    # (8)
                               [MAV.e3],    # (9)
                               [MAV.p0],    # (10)
                               [MAV.q0],    # (11)
                               [MAV.r0]])   # (12)
        # store wind data for fast recall since it is used at various points in simulation
        self._wind = np.array([[0.], [0.], [0.]])  # wind in NED frame in meters/sec
        self._update_velocity_data()
        # store forces to avoid recalculation in the sensors function
        self._forces = np.array([[0.], [0.], [0.]])
        self._Va = MAV.u0
        self._alpha = 0
        self._beta = 0
        # initialize true_state message
        self.msg_true_state = msg_state()

    ###################################
    # public functions
    def update_state(self, delta, wind):
        '''
            Integrate the differential equations defining dynamics, update sensors
            delta = (delta_a, delta_e, delta_r, delta_t) are the control inputs
            wind is the wind vector in inertial coordinates
            Ts is the time step between function calls.
        '''
        # get forces and moments acting on rigid bod
        forces_moments = self._forces_moments(delta)

        # Integrate ODE using Runge-Kutta RK4 algorithm
        time_step = self._ts_simulation
        k1 = self._derivatives(self._state, forces_moments)
        k2 = self._derivatives(self._state + time_step/2.*k1, forces_moments)
        k3 = self._derivatives(self._state + time_step/2.*k2, forces_moments)
        k4 = self._derivatives(self._state + time_step*k3, forces_moments)
        self._state += time_step/6 * (k1 + 2*k2 + 2*k3 + k4)

        # normalize the quaternion
        e0 = self._state.item(6)
        e1 = self._state.item(7)
        e2 = self._state.item(8)
        e3 = self._state.item(9)
        normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
        self._state[6][0] = self._state.item(6)/normE
        self._state[7][0] = self._state.item(7)/normE
        self._state[8][0] = self._state.item(8)/normE
        self._state[9][0] = self._state.item(9)/normE

        # update the airspeed, angle of attack, and side slip angles using new state
        self._update_velocity_data(wind)

        # update the message class for the true state
        self._update_msg_true_state()

    ###################################
    # private functions
    def _derivatives(self, state, forces_moments):
        """
        for the dynamics xdot = f(x, u), returns f(x, u)
        """
        # extract the states
        pn = state.item(0)
        pe = state.item(1)
        pd = state.item(2)
        u = state.item(3)
        v = state.item(4)
        w = state.item(5)
        e0 = state.item(6)
        e1 = state.item(7)
        e2 = state.item(8)
        e3 = state.item(9)
        p = state.item(10)
        q = state.item(11)
        r = state.item(12)
        #   extract forces/moments
        fx = forces_moments.item(0)
        fy = forces_moments.item(1)
        fz = forces_moments.item(2)
        l = forces_moments.item(3)
        m = forces_moments.item(4)
        n = forces_moments.item(5)

        # position kinematics
        pn_dot = (e1**2+e0**2-e2**2-e3**2)*u + 2*(e1*e2-e3*e0)*v + 2*(e1*e3+e2*e0)*w
        pe_dot = 2*(e1*e2+e3*e0)*u + (e2**2+e0**2-e1**2-e3**2)*v + 2*(e2*e3-e1*e0)*w
        pd_dot = 2*(e1*e3-e2*e0)*u + 2*(e2*e3+e1*e0)*v + (e3**2+e0**2-e1**2-e2**2)*w

        # position dynamics
        u_dot = r*v-q*w + fx/MAV.mass
        v_dot = p*w-r*u + fy/MAV.mass
        w_dot = q*u-p*v + fz/MAV.mass

        # rotational kinematics
        e0_dot = 0.5*(-p*e1-q*e2-r*e3)
        e1_dot = 0.5*(p*e0+r*e2-q*e3)
        e2_dot = 0.5*(q*e0-r*e1+p*e3)
        e3_dot = 0.5*(r*e0+q*e1-p*e2)

        # rotatonal dynamics
        p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + MAV.gamma3*l + MAV.gamma4*n
        q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2-r**2) + m/MAV.Jy
        r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + MAV.gamma4*l + MAV.gamma8*n

        # collect the derivative of the states
        x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                           e0_dot, e1_dot, e2_dot, e3_dot, p_dot, q_dot, r_dot]]).T
        return x_dot

    def _update_velocity_data(self, wind=np.zeros((6,1))):
        # compute airspeed
        ur = self._state[3] - wind[0]
        vr = self._state[4] - wind[1]
        wr = self._state[5] - wind[2]
        self._Va = sqrt(ur**2+vr**2+wr**2)
        # compute angle of attack
        self._alpha = atan2(wr,ur)
        # compute sideslip angle
        self._beta =asin(vr/sqrt(ur**2+vr**2+wr**2))

    def _forces_moments(self, delta):
        """
        return the forces on the UAV based on the state, wind, and control surfaces
        :param delta: np.matrix(delta_a, delta_e, delta_r, delta_t)
        :return: Forces and Moments on the UAV np.matrix(Fx, Fy, Fz, Ml, Mn, Mm)
        """
        # pull out neeed inputs
        p = self._state[10]
        q = self._state[11]
        r = self._state[12]
        delta_a = delta.item(0)
        delta_e = delta.item(1)
        delta_r = delta.item(2)
        delta_t = delta.item(3)

        # could convert straight from quaternions if I wanted to
        phi, theta, psi = Quaternion2Euler(self._state[6:10])

        M1 = np.matrix([[-MAV.mass*MAV.gravity*sin(theta)],
                        [MAV.mass*MAV.gravity*cos(theta)*sin(phi)],
                        [MAV.mass*MAV.gravity*sin(theta)*cos(phi)]])

        Tp = self.propThrust(delta_t)

        M2 = np.matrix([[Tp],
                       [0.],
                       [0.]])


        Cx = -self.CD(self._alpha)*cos(self._alpha) + self.CL(self._alpha)*sin(self._alpha)
        Cxq = -MAV.C_D_q*cos(self._alpha) + MAV.C_L_q*sin(self._alpha)
        Cxde = -MAV.C_D_delta_e*cos(self._alpha) + MAV.C_L_delta_e*sin(self._alpha)
        Cz = -self.CD(self._alpha)*sin(self._alpha) + self.CL(self._alpha)*cos(self._alpha)
        Czq = -MAV.C_D_q*sin(self._alpha) + MAV.C_L_q*cos(self._alpha)
        Czde = -MAV.C_D_delta_e*sin(self._alpha) + MAV.C_L_delta_e*cos(self._alpha)


        M3 = np.matrix([[float(Cx+Cxq*MAV.c*q/(2.*self._Va))],
                        [float(MAV.C_Y_0 + MAV.C_Y_beta*self._beta + MAV.C_Y_p*MAV.b*p/(2.*self._Va) + MAV.C_Y_r*MAV.b*r/(2*self._Va))],
                        [float(Cz + Czq*MAV.c*q/(2.*self._Va))]])

        M4 = np.matrix([[Cxde*delta_e],
                        [MAV.C_Y_delta_a*delta_a + MAV.C_Y_delta_r*delta_r],
                        [Czde*delta_e]])

        k1 = 0.5*MAV.rho*self._Va**2*MAV.S_wing

        force_result = M1 + M2 + k1*M3 + k1*M4

        fx = force_result.item(0)
        fy = force_result.item(1)
        fz = force_result.item(2)
        self._forces[0] = fx
        self._forces[1] = fy
        self._forces[2] = fz

        M5 = np.matrix([[float(MAV.b*(MAV.C_ell_0 + MAV.C_ell_beta*self._beta + MAV.C_ell_p*MAV.b*p/(2.*self._Va) + MAV.C_ell_r*MAV.b*r/(2.*self._Va)))],
                        [float(MAV.c*(MAV.C_m_0 + MAV.C_m_alpha*self._alpha + MAV.C_m_q*MAV.c*q/(2.*self._Va)))],
                        [float(MAV.b*(MAV.C_n_0 + MAV.C_n_beta*self._beta + MAV.C_n_p*MAV.b*p/(2.*self._Va) + MAV.C_n_r*MAV.b*r/(2.*self._Va)))]])

        M6 = np.matrix([[float(MAV.b*(MAV.C_ell_delta_a*delta_a + MAV.C_ell_delta_r*delta_r))],
                        [float(MAV.c*(MAV.C_m_delta_e*delta_e))],
                        [float(MAV.b*(MAV.C_n_delta_a*delta_a + MAV.C_n_delta_r*delta_r))]])

        Qp = self.propTorque(delta_t)

        M7 = np.matrix([[Qp],
                        [0.],
                        [0.]])

        moment_result = k1*M5 + k1*M6 + M7

        Mx = moment_result.item(0)
        My = moment_result.item(1)
        Mz = moment_result.item(2)

        return np.array([[fx, fy, fz, Mx, My, Mz]]).T

    def CD(self,alpha):
        '''
        UAV book equation 4.11
        '''
        result = MAV.C_D_p + ((MAV.C_L_0+alpha*MAV.C_L_alpha)**2)/(pi*MAV.e*MAV.AR)
        return result

    def CL(self,alpha):
        '''
        This is a linear coefficient model that is not valid over a wide
        range of angles of attack. UAV Book equation 4.13
        '''
        result = MAV.C_D_0 + MAV.C_D_alpha*alpha
        return result

    def propThrust(self,delta_t):

        Omega_op = self.propOperatingSpeed(delta_t)

        J_op = 2*pi*self._Va/(Omega_op*MAV.D_prop)
        Ct_op = MAV.C_T2*J_op**2 + MAV.C_T1*J_op + MAV.C_T0

        result = Ct_op*MAV.rho*(Omega_op**2)*(MAV.D_prop**4)/((2*pi)**2)
        return result

    def propOperatingSpeed(self,delta_t):
        a = MAV.rho*MAV.D_prop**5*MAV.C_Q0/(2*pi)**2
        b = MAV.rho*MAV.D_prop**4*MAV.C_Q1*self._Va/(2*pi) + MAV.KQ*MAV.K_V/MAV.R_motor
        c = MAV.rho*MAV.D_prop**3*MAV.C_Q2*self._Va**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0
        result = (-b + sqrt(b**2 - 4*a*c))/(2*a)
        return result

    def propTorque(self,delta_t):
        Vin = MAV.V_max*delta_t
        Omega_op = self.propOperatingSpeed(delta_t)
        result = MAV.KQ*((Vin - MAV.K_V*Omega_op)/MAV.R_motor - MAV.i0)
        return result

    def _update_msg_true_state(self):
        # update the class structure for the true state:
        #   [pn, pe, h, Va, alpha, beta, phi, theta, chi, p, q, r, Vg, wn, we, psi, gyro_bx, gyro_by, gyro_bz]


        Vg = sqrt(self._state[3]**2+self._state[4]**2+self._state[5]**2)
        gamma = atan2(self._state[5],sqrt(self._state[3]**2 + self._state[4]**2))
        chi = atan2(self._state[4],self._state[3])

        phi, theta, psi = Quaternion2Euler(self._state[6:10])
        self.msg_true_state.pn = self._state.item(0)
        self.msg_true_state.pe = self._state.item(1)
        self.msg_true_state.h = -self._state.item(2)
        self.msg_true_state.Va = self._Va
        self.msg_true_state.alpha = self._alpha
        self.msg_true_state.beta = self._beta
        self.msg_true_state.phi = phi
        self.msg_true_state.theta = theta
        self.msg_true_state.psi = psi
        self.msg_true_state.Vg = Vg
        self.msg_true_state.gamma = gamma
        self.msg_true_state.chi = chi
        self.msg_true_state.p = self._state.item(10)
        self.msg_true_state.q = self._state.item(11)
        self.msg_true_state.r = self._state.item(12)
        self.msg_true_state.wn = self._wind.item(0)
        self.msg_true_state.we = self._wind.item(1)
