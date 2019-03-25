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
from tools.tools import jacobian, wrapAnglePi2Pi #,Euler2Rotation

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
        self.position_ekf = ekf_position()

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
        self.position_ekf.update(self.estimated_state, measurements)

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
        self.Q = 1e-50*np.identity(2)
        self.Q_gyro = np.array([[SENSOR.gyro_sigma**2,0.,0.],
                                [0.,SENSOR.gyro_sigma**2,0.],
                                [0.,0.,SENSOR.gyro_sigma**2]])
        self.R_accel = np.array([[SENSOR.accel_sigma**2,0.,0.],
                                [0.,SENSOR.accel_sigma**2,0.],
                                [0.,0.,SENSOR.accel_sigma**2]])
        self.N = 4  # number of prediction step per sample
        self.xhat =  np.array([[P.phi0],[P.theta0]])# initial state: phi, theta
        self.P = np.array([[1.,0.],
                           [0.,1.]])
        self.Ts = SIM.ts_control/self.N

    def update(self, state, measurement):
        self.propagate_model(state)
        #self.measurement_update(state, measurement)
        state.phi = self.xhat.item(0)
        state.theta = self.xhat.item(1)

    def f(self, x, state):
        # system dynamics for propagation model: xdot = f(x, u)
        phi = x.item(0)
        theta = x.item(1)
        G = np.array([[1.0, np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],
                      [0., np.cos(phi), -np.sin(phi)]])
        _f = G @ np.array([[state.p],
                           [state.q],
                           [state.r]])
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
        for ii in range(self.N):
            # propagate model
            #x = np.array([[state.phi,state.theta]])
            self.xhat += self.Ts*self.f(self.xhat,state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # compute G matrix for gyro noise
            G = np.array([[1.0, np.sin(state.phi)*np.tan(state.theta),np.cos(state.phi)*np.tan(state.theta)],
                          [0., np.cos(state.phi), -np.sin(state.phi)]])
            # update P with continuous time model
            self.P += self.Ts * (A @ self.P + self.P @ A.T + self.Q + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            #A_d =
            #G_d = ts*G
            # update P with discrete time model
            #self.P =

    def measurement_update(self, state, measurement):
        # measurement updates
        threshold = 2.0
        h = self.h(self.xhat, state)
        C = jacobian(self.h, self.xhat, state)
        y = np.array([[measurement.accel_x], [measurement.accel_y], [measurement.accel_z]])
        error = False
        for ii in range(3):
            if np.abs(y[ii]-h[ii,0]) > threshold:
                error = True
        if not(error):
            Ci = C
            Li = self.P @ Ci.T @ np.linalg.inv(self.R_accel + Ci @ self.P @ Ci.T)
            self.P = (np.identity(2) - Li @ Ci) @ self.P
            self.xhat += Li @ (y - h)

class ekf_position:
    # implement continous-discrete EKF to estimate pn, pe, chi, Vg
    def __init__(self):
        self.Q = np.diag([1e-50,
                          1e-50,
                          1e-1,
                          1e-50,
                          5e-2,
                          5e-2,
                          1e-1])
        self.R = np.array([[SENSOR.gps_n_sigma**2,0.,0.,0.],
                           [0.,SENSOR.gps_e_sigma**2,0.,0.],
                           [0.,0.,SENSOR.gps_Vg_sigma**2,0.],
                           [0.,0.,0.,SENSOR.gps_course_sigma**2]])
        self.N = 4  # number of prediction step per sample
        self.Ts = (SIM.ts_control / self.N)
        self.xhat = np.array([[P.pn0],
                              [P.pe0],
                              [P.u0],
                              [P.psi0],
                              [0.0],
                              [0.0],
                              [P.psi0]])
        self.P = np.identity(7)
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
        Vg = x.item(2)
        Va = state.Va
        chi = x.item(3)
        psi = x.item(6)
        phi = state.phi
        theta = state.theta
        q = state.q
        r = state.r
        psi_dot = q*np.sin(phi)/np.cos(theta)+r*np.cos(phi)/np.cos(theta)
        wn = x.item(4)
        we = x.item(5)

        _f = np.array([[Vg*np.cos(chi)],
                       [Vg*np.sin(chi)],
                       [((Va*np.cos(psi)+wn)*(-Va*psi_dot*np.sin(psi))+(Va*np.sin(psi)+we)*(Va*psi_dot*np.cos(psi)))/Vg],
                       [P.gravity*np.tan(phi)*np.cos(chi-psi)/Vg],
                       [0.1],
                       [0.1],
                       [psi_dot]])
        return _f

    def h_gps(self, x, state):
        # measurement model for gps measurements
        pn = x.item(0)
        pe = x.item(1)
        Vg = x.item(2)
        chi = x.item(3)

        _h = np.array([[pn],
                       [pe],
                       [Vg],
                       [chi]])
        return _h

    def h_pseudo(self, x, state):
        # measurement model for wind triangale pseudo measurement
        Va = state.Va
        psi = x.item(6)
        wn = x.item(4)
        we = x.item(5)
        Vg = x.item(2)
        chi = x.item(3)
        _h = np.array([[Va*np.cos(psi)+wn-Vg*np.cos(chi)],
                       [Va*np.sin(psi)+we-Vg*np.sin(chi)]])
        return _h

    def propagate_model(self, state):
        # model propagation
        for i in range(0, self.N):
            # propagate model
            self.xhat += self.Ts*self.f(self.xhat,state)
            # compute Jacobian
            A = jacobian(self.f, self.xhat, state)
            # update P with continuous time model
            self.P += self.Ts * (A @ self.P + self.P @ A.T + self.Q)# + G @ self.Q_gyro @ G.T)
            # convert to discrete time models
            #A_d =
            # update P with discrete time model
            #self.P =

    def measurement_update(self, state, measurement):
        # always update based on wind triangle pseudu measurement
        h = self.h_pseudo(self.xhat, state)
        C = jacobian(self.h_pseudo, self.xhat, state)
        y = np.array([[0.],[0.]])
        threshold = 10.0
        error = False
        for ii in range(2):
            if np.abs(y[ii]-h[ii,0]) > threshold:
                error = True
        if not(error):
            Ci = C[:,4:6]
            Li = self.P[4:6,4:6] @ Ci.T
            self.P[4:6,4:6] = (np.identity(2) - Li @ Ci) @ self.P[4:6,4:6]
            self.xhat[4:6] += Li @ (y-h)

        # only update GPS when one of the signals changes
        if (measurement.gps_n != self.gps_n_old) \
            or (measurement.gps_e != self.gps_e_old) \
            or (measurement.gps_Vg != self.gps_Vg_old) \
            or (measurement.gps_course != self.gps_course_old):

            h = self.h_gps(self.xhat, state)
            C = jacobian(self.h_gps, self.xhat, state)
            y = np.array([[measurement.gps_n], [measurement.gps_e], [measurement.gps_Vg], [measurement.gps_course]])
            Ci = C[:,0:4]
            Li = self.P[0:4,0:4] @ Ci.T @ np.linalg.inv(self.R + Ci @ self.P[0:4,0:4] @ Ci.T)
            self.P[0:4,0:4] = (np.identity(4)-Li*Ci) @ self.P[0:4,0:4]
            new_add = Li @ (y - h)
            threshold = np.array([20.0,30.0,10.0,np.radians(5.0)])
            for ii in range(4):
                if np.abs(new_add.item(ii)) < threshold[ii]:
                    self.xhat[ii] += new_add.item(ii)
                else:
                    pass
                    #print(ii,new_add.item(ii))
            # update stored GPS signals
            self.gps_n_old = measurement.gps_n
            self.gps_e_old = measurement.gps_e
            self.gps_Vg_old = measurement.gps_Vg
            self.gps_course_old = measurement.gps_course

            self.xhat[3] = wrapAnglePi2Pi(self.xhat[3])
            self.xhat[6] = wrapAnglePi2Pi(self.xhat[6])
