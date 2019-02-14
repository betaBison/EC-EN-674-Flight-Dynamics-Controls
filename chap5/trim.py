"""
compute_trim
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion
import parameters.aerosonde_parameters as MAV
from chap4.mav_dynamics import mav_dynamics as dyn

def compute_trim(mav, Va, gamma):
    phi = 0.0
    theta = gamma
    psi = 0.0
    e0,e1,e2,e3 = Euler2Quaternion(phi, theta, psi)
    # define initial state and input
    state0 = np.array([[0.0],  # (0)
                       [0.0],   # (1)
                       [0.0],   # (2)
                       [20.0],    # u0
                       [0.0],    # v0
                       [0.0],    # w0
                       [e0],    # e0
                       [e1],    # e1
                       [e2],    # e2
                       [e3],    # e3
                       [0.0],    # p0
                       [0.0],    # q0
                       [0.0]])   # r0
    delta_a = 0.0
    delta_e = 0.1
    delta_r = 0.0
    delta_t = 0.5
    delta0 = np.array([[delta_a, delta_e, delta_r, delta_t]]).T
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma),
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T
    return trim_state, trim_input

def calcX_dot_star(Va,gamma):
    gamma_star = gamma
    Va_star = Va
    x_dot_star = np.array([[0.0],  # pn_dot don't care
                           [0.0],   # pe_dot don't care
                           [Va_star*np.sin(gamma_star)], # h_dot
                           [0.0],    # u_dot
                           [0.0],    # v_dot
                           [0.0],    # w_dot
                           [0.0],    # phi_dot
                           [0.0],    # theta_dot
                           [0.0],    # psi_dot Va_star*np.cos(gamma_star)/R_star
                           [0.0],
                           [0.0],    # p_dot
                           [0.0],    # q_dot
                           [0.0]])   # r_dot
    return x_dot_star

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):
    print("x=",x)
    x_star = x[0:13]
    e0 = x_star[6]
    e1 = x_star[7]
    e2 = x_star[8]
    e3 = x_star[9]
    normE = np.sqrt(e0**2+e1**2+e2**2+e3**2)
    e0 /= normE
    e1 /= normE
    e2 /= normE
    e3 /= normE
    x_star[6] = e0
    x_star[7] = e1
    x_star[8] = e2
    x_star[9] = e3
    delta_star = x[13:17]
    mav._state = x_star
    mav._update_velocity_data()
    Va_star = Va
    gamma_star = gamma
    x_dot_star = calcX_dot_star(Va_star,gamma_star)
    print("Va in objective",Va)
    print("Va in state=",mav._Va)
    print("delta star",delta_star)
    force_moments = mav._forces_moments(delta_star)
    f_result = mav._derivatives(x_star,force_moments)

    J = np.linalg.norm(x_dot_star[2:13] - f_result[2:13])**2
    return J
