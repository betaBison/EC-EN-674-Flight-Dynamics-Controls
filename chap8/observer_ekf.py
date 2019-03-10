"""
observer
    - Beard & McLain, PUP, 2012
    - Last Update:
        3/2/2019 - RWB
"""
import sys
import numpy as np
sys.path.append('..')
import parameters.control_parameters as CTRL
import parameters.aerosonde_parameters as P
import parameters.simulation_parameters as SIM
import parameters.sensor_parameters as SENSOR
from tools.tools import jacobian #,Euler2Rotation

from message_types.msg_state import msg_state

class observer:
    def __init__(self, ts_control):
        # initialized estimated state message
        self.estimated_state = msg_state()
        # use alpha filters to low pass filter gyros and accels
        self.lpf_gyro_x = alpha_filter(alpha=0.5)
        self.lpf_gyro_y = alpha_filter(alpha=0.5)
        self.lpf_gyro_z = alpha_filter(alpha=0.5)
        self.lpf_accel_x = alpha_filter(alpha=0.9)
        self.lpf_accel_y = alpha_filter(alpha=0.9)
        self.lpf_accel_z = alpha_filter(alpha=0.9)
        # use alpha filters to low pass filter static and differential pressure
        self.lpf_static = alpha_filter(alpha=0.9)
        self.lpf_diff = alpha_filter(alpha=0.5)


        # ekf for phi and theta
        self.attitude_ekf = ekf_attitude()
        # ekf for pn, pe, Vg, chi, wn, we, psi
        #self.position_ekf = ekf_position()

    def update(self, measurements):

        # estimates for p, q, r are low pass filter of gyro minus bias estimate
        self.estimated_state.p = self.lpf_gyro_x.update(measurements.gyro_x) - SENSOR.gyro_x_bias
        self.estimated_state.q = self.lpf_gyro_y.update(measurements.gyro_y) - SENSOR.gyro_y_bias
        self.estimated_state.r = self.lpf_gyro_z.update(measurements.gyro_z) - SENSOR.gyro_z_bias

        # invert sensor model to get altitude and airspeed
        self.estimated_state.h = self.lpf_static.update(measurements.static_pressure)/(P.rho*P.gravity)
        self.estimated_state.Va = np.sqrt((2./P.rho)*self.lpf_diff.update(measurements.diff_pressure))

        # estimate phi and theta with simple ekf
        self.attitude_ekf.update(self.estimated_state, measurements)

        # estimate pn, pe, Vg, chi, wn, we, psi
        #self.position_ekf.update(self.estimated_state, measurements)

        # not estimating these
        self.estimated_state.alpha = self.estimated_state.theta
        self.estimated_state.beta = 0.0
        self.estimated_state.bx = 0.0
        self.estimated_state.by = 0.0
        self.estimated_state.bz = 0.0
        return self.estimated_state


class alpha_filter:
    # alpha filter implements a simple low pass filter
    # y[k] = alpha * y[k-1] + (1-alpha) * u[k]
    def __init__(self, alpha=0.5, y0=0.0):
        self.alpha = alpha  # filter parameter
        self.y = y0  # initial condition

    def update(self, u):
        self.y *= self.alpha
        self.y += (1.-self.alpha)*u
        return self.y

class ekf_attitude:
    # implement continous-discrete EKF to estimate roll and pitch angles
    def __init__(self):
        self.Q = 1e-5*np.identity(3)
        self.Q_gyro = np.array([[SENSOR.gyro_sigma**2,0.,0.],
                                [0.,SENSOR.gyro_sigma**2,0.],
                                [0.,0.,SENSOR.gyro_sigma**2]])
        self.R_accel = 0
        self.N = 1.  # number of prediction step per sample
        self.xhat =  np.array([P.phi0,P.theta0])# initial state: phi, theta
        self.P = 0
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x.item(0)
        theta = x.item(1)
        G = np.array([[]])
        _f = G @ np.array()
        return _f

    def h(self, x, state):
        # measurement model y
        phi = x.item(0)
        theta = x.item(1)
        _h = np.array([[state.q*state.Va*np.sin(theta) + P.gravity*np.sin(theta)],
                       [state.r*state.Va*np.cos(theta) - state.p*state.Va*np.sin(theta) - P.gravity*np.cos(theta)*np.sin(phi)],
                       [-state.q*state.Va*np.cos(theta) - P.gravity*np.cos(theta)*np.cos(phi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
             # propagate model
            self.xhat = np.array([state.phi,state.theta])
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G =
            # update P with continuous time model
            self.P += self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            #A_d =
            #G_d =
            # update P with discrete time model
            #self.P =

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([measurement.accel_x, measurement.accel_y, measurement.accel_z])
        for i in range(0, 3):
            if np.abs(y[i]-h[i,0]) < threshold:
                Ci = C.item(ii)
                Li = self.P @ Ci @ np.linalg.inv(self.R_accel + Ci @ self.P @ Ci.T)
                self.P = (np.identity(3) - Li @ Ci) @ self.P
                self.xhat += Li @ (y.item(ii) - h.item(ii))
'''
class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        self.Q =
        self.R =
        self.N =   # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat =
        self.P =
        self.gps_n_old = 9999
        self.gps_e_old = 9999
        self.gps_Vg_old = 9999
        self.gps_course_old = 9999


    def update(self, state, measurement):
        self.propagate_model(state)
        self.measurement_update(state, measurement)
        state.pn = self.xhat.item(0)
        state.pe = self.xhat.item(1)
        state.Vg = self.xhat.item(2)
        state.chi = self.xhat.item(3)
        state.wn = self.xhat.item(4)
        state.we = self.xhat.item(5)
        state.psi = self.xhat.item(6)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        _f =
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        _h =
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangale pseudo measurement
        _h =
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat =
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            # self.P = self.P + self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            A_d =
            # update P with discrete time model
            self.P =

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudu measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([0, 0])
        for i in range(0, 2):
            Ci =
            L =
            self.P =
            self.xhat =

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([measurement.gps_n, measurement.gps_e, measurement.gps_Vg, measurement.gps_course])
            for i in range(0, 4):
                Ci =
                L =
                self.P =
                self.xhat =
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

    def wrap(self, chi_c, chi):
        while chi_c-chi > np.pi:
            chi_c = chi_c - 2.0 * np.pi
        while chi_c-chi < -np.pi:
            chi_c = chi_c + 2.0 * np.pi
        return chi_c
'''
