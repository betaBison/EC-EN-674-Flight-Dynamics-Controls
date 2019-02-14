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
    # define initial state and input
    state0 = np.array([[0.0],  # (0)
                       [0.0],   # (1)
                       [-100.0],   # (2)
                       [25.0],    # (3)
                       [0.0],    # (4)
                       [0.1],    # (5)
                       [1.0],    # (6)
                       [0.0],    # (7)
                       [0.0],    # (8)
                       [0.0],    # (9)
                       [0.0],    # (10)
                       [0.0],    # (11)
                       [0.0]])   # (12)
    delta_a = 0.005
    delta_e = -0.2
    delta_r = 0.008
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

def calcX_dot_star(Va_star,gamma_star):
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
                           [0.0],    # p_dot
                           [0.0],    # q_dot
                           [0.0]])   # r_dot
    return x_dot_star

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):
    Va_star = Va
    gamma_star = gamma
    #
    phi_star = 0.0
    theta = 0.0
    alpha_star = theta-gamma
    chi = 0.0
    psi = 0.0
    beta_star = chi-psi

    x_dot_star = calcX_dot_star(Va,gamma,R_star)
    u_star = Va_star*np.cos(alpha_star)*np.cos(beta_star)
    v_star = Va_star*np.sin(beta_star)
    w_star = Va_star*np.sin(alpha_star)*np.cos(beta_star)
    theta_star = alpha_star + gamma_star
    p_star = 0.0 #-Va_star*np.sin(theta_star)/R_star
    q_star = 0.0 #Va_star*np.sin(phi_star)*np.cos(theta_star)/R_star
    r_star = 0.0 #Va_star*np.cos(phi_star)*np.sin(theta_star)/R_star
    # F.1
    delta_e_star = (((MAV.Jxz*(p_star**2-r_star**2)+(MAV.Jx-MAV.Jz)*p_star*r_star)/(0.5*MAV.rho*Va_star**2*MAV.c*MAV.S_wing))-MAV.C_m_0-MAV.C_m_alpha*alpha_star-MAV.C_m_q*MAV.c*q_star/(2.*Va_star))

    # F.2
    Cx = -dyn.CD(alpha_star)*np.cos(alpha_star) + dyn.CL(alpha_star)*np.sin(alpha_star)
    Cxq = -MAV.C_D_q*np.cos(alpha_star) + MAV.C_L_q*np.sin(alpha_star)
    Cxde = -MAV.C_D_delta_e*np.cos(alpha_star) + MAV.C_L_delta_e*np.sin(alpha_star)

    delta_t_star = np.sqrt((2.*MAV.mass*(-r_star*v_star+q_star*w_star+MAV.gravity*np.sin(theta_star))
        -MAV.rho*Va_star**2*MAV.S_wing*(Cx+Cxq*MAV.c*q_star/(2.*Va_star)+Cxde*delta_e_star))
        /(MAV.rho*MAV.S_prop*MAV.C_prop*MAV.k_motor**2)+Va_star**2/MAV.k_motor**2)

    # F.3
    A = np.matrix([[MAV.C_p_delta_a,MAV.C_p_delta_r],
                  [MAV.C_r_delta_a,MAV.C_r_delta_r]])
    B = np.matrix([[(-MAV.gamma1*p_star*q_star+MAV.gamma2*q_star*r_star)/(0.5*MAV.rho*Va_star**2*MAV.S_wing*MAV.b)
                    -MAV.C_p_0-MAV.C_p_beta*beta_star-MAV.C_p_p*MAV.b*p_star/(2.*Va_star)-MAV.C_p_r*MAV.b*r_star/(2.*Va_star)],
                   [(-MAV.gamma7*p_star*q_star+MAV.gamma1*q_star*r_star)/(0.5*MAV.rho*Va_star**2*MAV.S_wing*MAV.b)
                    -MAV.C_r_0-MAV.C_r_beta*beta_star-MAV.C_r_p*MAV.b*p_star/(2.*Va_star)-MAV.C_r_r*MAV.b*r_star/(2.*Va_star)]])
    delta_result = np.cross(np.linalg.inv(A),B)
    delta_a_star = delta_result.item(0)
    delta_r_star = relta_result.item(1)

    delta = np.array([[delta_a_star, delta_e_star, delta_r_star, delta_t_star]]).T

    f_result = calcLinearDerivatives(x_dot_star,delta,alpha_star,Va_star,beta_star)

    J = np.linalg.norm(x_dot_star - f_result)**2
    return J

def calcLinearDerivatives(x_star,u_star,alpha,Va,beta):
    """
    for the dynamics xdot = f(x, u), returns f(x, u)
    """
    # extract the states
    pn = x_star.item(0)
    pe = x_star.item(1)
    pd = x_star.item(2)
    u = x_star.item(3)
    v = x_star.item(4)
    w = x_star.item(5)
    phi = x_star.item(6)
    theta = x_star.item(7)
    psi = x_star.item(8)
    p = x_star.item(9)
    q = x_star.item(10)
    r = x_star.item(11)

    #  inputs
    delta_a = u_star.item(0)
    delta_e = u_star.item(1)
    delta_r = u_star.item(2)
    delta_t = u_star.item(3)


    # position kinematics
    pn_dot = (np.cos(theta)*np.cos(psi))*u + (np.sin(phi)*np.sin(theta)*np.cos(psi)-np.cos(phi)*np.sin(psi))*v + (np.cos(phi)*np.sin(theta)*np.cos(psi)+np.sin(phi)*np.sin(psi))*w
    pe_dot = (np.cos(theta)*np.sin(psi))*u + (np.sin(phi)*np.sin(theta)*np.sin(psi)+np.cos(phi)*np.cos(psi))*v + (np.cos(phi)*np.sin(theta)*np.sin(psi)-np.sin(phi)*np.cos(psi))*w
    pd_dot = -(np.sin(theta)*u-np.sin(phi)*np.cos(theta)*v-np.cos(phi)*np.cos(theta)*w)



    # position dynamics
    Cx = -CD(alpha)*cos(alpha) + CL(alpha)*sin(alpha)
    Cxq = -MAV.C_D_q*cos(alpha) + MAV.C_L_q*sin(alpha)
    Cxde = -MAV.C_D_delta_e*cos(alpha) + MAV.C_L_delta_e*sin(alpha)
    Cz = -CD(alpha)*sin(alpha) - CL(alpha)*cos(alpha)
    Czq = -MAV.C_D_q*sin(alpha) - MAV.C_L_q*cos(alpha)
    Czde = -MAV.C_D_delta_e*sin(alpha) - MAV.C_L_delta_e*cos(alpha)

    u_dot = r*v-q*w + MAV.gravity*np.sin(theta) + MAV.rho*Va**2*MAV.S_wing*(Cx+Cxq*MAV.c*q/(2.*Va)+Cxde*delta_e)/(2.*MAV.mass) +
        MAV.rho*MAV.S_prop*MAV.C_prop*((MAV.k_motor*delta_t)**2-Va*2)/(2.*MAV.mass)
    v_dot = p*w-r*u + MAV.gravity*np.cos(theta)*np.sin(phi)+
        MAV.rho*Va**2*MAV.S_wing*(MAV.C_Y_0+MAV.C_Y_beta*beta*MAV.C_Y_p*MAV.b*p/(2.*Va)+MAV.C_Y_r*MAV.b*r/(2.*Va)+MAV.C_Y_delta_a*delta_a+MAV.C_Y_delta_r*delta_r))/(2.*MAV.mass)
    w_dot = q*u-p*v + MAV.gravity*np.cos(theta)*np.cos(phi)+MAV.rho*Va**2*MAV.S_wing*(Cz+Czq*MAV.c*q/(2.*Va)+Czde*delta_e)/(2.*MAV.mass)

    # rotational kinematics
    phi_dot = p + q*np.sin(phi)*np.tan(theta)+r*np.cos(phi)*np.tan(theta)
    theta_dot = q*np.cos(phi)-r*np.sin(phi)
    psi_dot = q*np.sin(phi)/np.cos(theta)+r*np.cos(phi)/np.cos(theta)

    # rotatonal dynamics
    p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r +
        0.5*MAV.rho*Va**2*MAV.S_wing*MAV.b*(MAV.C_p_0+MAV.C_p_beta*beta*MAV.C_p_p*MAV.b*p/(2.*Va)+MAV.C_p_r*MAV.b*r/(2.*Va)+MAV.C_p_delta_a*delta_a+MAV.C_p_delta_r*delta_r)
    q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2-r**2) + MAV.rho*Va**2*MAV.S_wing*MAV.c*(MAV.C_m_0+MAV.C_m_alpha*alpha+MAV.C_m_q*MAV.c*q/(2.*Va)+MAV.C_m_delta_e*delta_e)/(2.*MAV.Jy)
    r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r +
        0.5*MAV.rho*Va**2*MAV.S_wing*MAV.b*(MAV.C_r_0+MAV.C_r_beta*beta*MAV.C_r_p*MAV.b*p/(2.*Va)+MAV.C_r_r*MAV.b*r/(2.*Va)+MAV.C_r_delta_a*delta_a+MAV.C_r_delta_r*delta_r)

    # collect the derivative of the states
    x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                       phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]]).T
    return x_dot

def CD(alpha):
    '''
    UAV book equation 4.11
    '''
    result = MAV.C_D_p + ((MAV.C_L_0+alpha*MAV.C_L_alpha)**2)/(pi*MAV.e*MAV.AR)
    #result = (1-calcSigma(alpha))*(MAV.C_D_0 + MAV.C_D_alpha*alpha)+calcSigma(alpha)*(2.0*np.sign(alpha)*sin(alpha)**2*cos(alpha))
    return result

def CL(alpha):
    '''
    This is a linear coefficient model that is not valid over a wide
    range of angles of attack. UAV Book equation 4.13
    '''
    result = (1-calcSigma(alpha))*(MAV.C_L_0 + MAV.C_L_alpha*alpha)+calcSigma(alpha)*(2.0*np.sign(alpha)*sin(alpha)**2*cos(alpha))
    return result

def calcSigma(alpha):
    # blending function according to ch 4 UAV book slides
    nom = 1.0 + np.exp(-MAV.M*(alpha-MAV.alpha0))+np.exp(MAV.M*(alpha+MAV.alpha0))
    den = (1.0 + np.exp(-MAV.M*(alpha-MAV.alpha0)))*(1+np.exp(MAV.M*(alpha+MAV.alpha0)))
    result = nom/den
    return result
