"""
compute_ss_model
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:
        2/4/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion, Quaternion2Euler
#from tools.transfer_function import transfer_function
import parameters.aerosonde_parameters as MAV
from parameters.simulation_parameters import ts_simulation as Ts
import control.matlab as ctrl

def compute_tf_model(mav, trim_state, trim_input):
    # trim values

    mav._state = trim_state
    mav._update_velocity_data()

    state_euler = euler_state(trim_state)
    #print(trim_state)
    #print(state_euler)
    # extract the states
    pn = state_euler.item(0)
    pe = state_euler.item(1)
    pd = state_euler.item(2)
    u = state_euler.item(3)
    v = state_euler.item(4)
    w = state_euler.item(5)
    phi = state_euler.item(6)
    theta = state_euler.item(7)
    psi = state_euler.item(8)
    p = state_euler.item(9)
    q = state_euler.item(10)
    r = state_euler.item(11)
    #  inputs
    delta_a = trim_input.item(0)
    delta_e = trim_input.item(1)
    delta_r = trim_input.item(2)
    delta_t = trim_input.item(3)


    # Transfer function delta_a --> phi
    a_phi_1 = -0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2.*mav._Va)
    a_phi_2 = 0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a
    #d_phi_1 = q*np.sin(phi)*np.tan(theta) + r*np.cos(phi)*np.tan(theta)
    #d_dot_phi_1 = 0.0 #help
    #d_phi_2 = MAV.gamma1*p*q - MAV.gamma2*q*r + \
    #    0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*(MAV.C_p_0+MAV.C_p_beta*mav._beta-MAV.C_p_p*MAV.b*d_phi_1/(2.*mav._Va)+MAV.C_p_r*MAV.b*r/(2.*mav._Va)+MAV.C_p_delta_r*delta_r) + d_dot_phi_1
    T_phi_delta_a = ctrl.tf([a_phi_2],[1,a_phi_1,0])
    print("T_phi_delta_a",T_phi_delta_a)

    # Transfer function phi --> chi
    #dx = np.tan(phi)-phi
    T_chi_phi = ctrl.tf([MAV.gravity/mav._Vg],[1,0])
    print("T_chi_phi",T_chi_phi)


    # Transfer function delta_e --> theta
    a_theta_1 = -MAV.rho*mav._Va**2*MAV.c*MAV.S_wing*MAV.C_m_q*MAV.c/(2.*MAV.Jy*2.*mav._Va)
    a_theta_2 = -MAV.rho*mav._Va**2*MAV.c*MAV.S_wing*MAV.C_m_alpha/(2.*MAV.Jy)
    a_theta_3 = MAV.rho*mav._Va**2*MAV.c*MAV.S_wing*MAV.C_m_delta_e/(2.*MAV.Jy)
    #d_theta_2 = MAV.gamma6*(r**2-p**2) + MAV.gamma5*p*r + MAV.rho*mav._Va**2*MAV.S_wing*MAV.c*(MAV.C_m_0-MAV.C_m_alpha*mav.gamma-MAV.C_m_q*MAV.c*a_theta_1)/(2.*Va)+MAV.C_m_delta_e*delta_e)/(2.*MAV.Jy)
    T_theta_delta_e = ctrl.tf([a_theta_3],[1,a_theta_1,a_theta_2])
    print("T_theta_delta_e",T_theta_delta_e )

    # Transfer function theta --> h
    T_h_theta = ctrl.tf([mav._Va],[1,0])
    print("T_h_theta",T_h_theta)

    # Transfer function Va --> h
    T_h_Va = ctrl.tf([theta],[1,0])
    print("T_h_Va",T_h_Va)

    # Transfer function delta_t --> Va
    a_v_1 = MAV.rho*mav._Va*MAV.S_wing*(MAV.C_D_0 + MAV.C_D_alpha*mav._alpha+MAV.C_D_delta_e*delta_e)/MAV.mass + MAV.rho*MAV.S_prop*MAV.C_prop*mav._Va/MAV.mass
    a_v_2 = MAV.rho*MAV.S_prop*MAV.C_prop*MAV.k_motor**2*delta_t/MAV.mass
    T_Va_delta_t = ctrl.tf([a_v_2],[1.0,a_v_1])
    print("T_Va_delta_t",T_Va_delta_t)

    # Transfer funciton theta --> Va
    a_v_3 = MAV.gravity
    T_Va_theta = ctrl.tf([-a_v_3],[1.0,a_v_1])
    print("T_Va_theta",T_Va_theta)

    # Transfer function delta_r --> beta
    a_beta_1 = -MAV.rho*mav._Va*MAV.S_wing*MAV.C_Y_beta/(2.*MAV.mass)
    a_beta_2 = MAV.rho*mav._Va*MAV.S_wing*MAV.C_Y_delta_r/(2.*MAV.mass)
    #d_beta = p*w-r*u + MAV.gravity*np.cos(theta)*np.sin(phi)+ \
    #    MAV.rho*mav._Va**2*MAV.S_wing*(MAV.C_Y_0+MAV.C_Y_p*MAV.b*p/(2.*mav._Va)+MAV.C_Y_r*MAV.b*r/(2.*mav._Va)+MAV.C_Y_delta_a*delta_a)/(2.*MAV.mass)
    T_beta_delta_r = ctrl.tf([a_beta_2],[1.0,a_beta_1])
    print("T_beta_delta_r",T_beta_delta_r)

    return T_phi_delta_a, T_chi_phi, T_theta_delta_e, T_h_theta, T_h_Va, T_Va_delta_t, T_Va_theta, T_beta_delta_r

def compute_ss_model(mav, trim_state, trim_input):
    mav._state = trim_state
    mav._update_velocity_data()

    state_euler = euler_state(trim_state)
    print(trim_state)
    print(state_euler)
    # extract the states
    pn = state_euler.item(0)
    pe = state_euler.item(1)
    pd = state_euler.item(2)
    u = state_euler.item(3)
    v = state_euler.item(4)
    w = state_euler.item(5)
    phi = state_euler.item(6)
    theta = state_euler.item(7)
    psi = state_euler.item(8)
    p = state_euler.item(9)
    q = state_euler.item(10)
    r = state_euler.item(11)
    #  inputs
    delta_a = trim_input.item(0)
    delta_e = trim_input.item(1)
    delta_r = trim_input.item(2)
    delta_t = trim_input.item(3)



    beta_star = mav._beta
    Va_star = mav._Va

    u_star = u
    v_star = v
    w_star = w
    phi_star = phi
    theta_star = theta
    p_star = p
    q_star = q
    r_star = r

    delta_a_star = delta_a
    delta_r_star = delta_r

    Cx = -CD(mav._alpha)*np.cos(mav._alpha) + CL(mav._alpha)*np.sin(mav._alpha)
    Cxq = -MAV.C_D_q*np.cos(mav._alpha) + MAV.C_L_q*np.sin(mav._alpha)
    Cxde = -MAV.C_D_delta_e*np.cos(mav._alpha) + MAV.C_L_delta_e*np.sin(mav._alpha)
    Cz = -CD(mav._alpha)*np.sin(mav._alpha) - CL(mav._alpha)*np.cos(mav._alpha)
    Czq = -MAV.C_D_q*np.sin(mav._alpha) - MAV.C_L_q*np.cos(mav._alpha)
    Czde = -MAV.C_D_delta_e*np.sin(mav._alpha) - MAV.C_L_delta_e*np.cos(mav._alpha)

    Cx0 = Cx

    Xu = u_star*MAV.rho*MAV.S_wing*(Cx0)
    Xw = 0.0
    Xq = 0.0
    X_delta_e = 0.0
    X_delta_t = 0.0
    Zu = 0.0
    Zw = 0.0
    Zq = 0.0
    Z_delta_e = 0.0
    Mu = 0.0
    Mw = 0.0
    Mq = 0.0
    M_delta_e = 0.0


    A_lon = np.array([[Xu, Xw*Va_star*np.cos(mav._alpha),Xq,-MAV.gravity*np.cos(theta_star),0.0],
                      [Zu/(Va_star*np.cos(mav._alpha)),Zw,Zq/(Va_star*np.cos(mav._alpha)),-MAV.gravity*np.sin(theta_star)/(Va_star*np.cos(mav._alpha)),0.0],
                      [Mu,Mw*Va_star,np.cos(mav._alpha),Mq,0.0,0.0],
                      [0.0,0.0,1.0,0.0,0.0],
                      [np.sin(theta_star),-Va_star*np.cos(theta_star)*np.cos(mav._alpha),0.0,u_star*np.cos(theta_star)+w_star*np.sin(theta_star),0.0]])
    B_lon = np.array([[X_delta_e,X_delta_t],
                      [Z_delta_e/(Va_star*np.cos(mav._alpha)),0.0],
                      [M_delta_e,0.0],
                      [0.0,0.0],
                      [0.0,0.0]])


    #lon_eig = np.linalg.eig(A_lon)
    #print(lon_eig)

    # QUESTION: m in Yv equation supposed to be mass?
    Yv = MAV.rho*MAV.S_wing*MAV.b*v_star*(MAV.C_Y_p*p_star+MAV.C_Y_r*r_star)/(4.*MAV.mass*Va_star) \
        + MAV.rho*MAV.S_wing*v_star*(MAV.C_Y_0+MAV.C_Y_beta*beta_star+MAV.C_Y_delta_a*delta_a_star+MAV.C_Y_delta_r*delta_r_star)/MAV.mass \
        + MAV.rho*MAV.S_wing*MAV.C_Y_beta*np.sqrt(u_star**2+w_star**2)/(2.*MAV.mass)
    # QUESTION: m in Yp equation supposed to be mass?
    Yp = w_star + MAV.rho*Va_star*MAV.S_wing*MAV.b*MAV.C_Y_p/(4.*MAV.mass)
    # QUESTION: m in Yr equation supposed to be mass?
    Yr = -u_star + MAV.rho*Va_star*MAV.S_wing*MAV.b*MAV.C_Y_r/(4.*MAV.mass)
    Y_delta_a = MAV.rho*Va_star**2*MAV.S_wing*MAV.C_Y_delta_a/(2.*MAV.mass)
    Y_delta_r = MAV.rho*Va_star**2*MAV.S_wing*MAV.C_Y_delta_r/(2.*MAV.mass)
    Lv = MAV.rho*MAV.S_wing*MAV.b**2*v_star*(MAV.C_p_p*p_star+MAV.C_p_r*r_star)/(4.*Va_star) \
        + MAV.rho*MAV.S_wing*MAV.b*v_star*(MAV.C_p_0+MAV.C_p_beta*beta_star+MAV.C_p_delta_a*delta_a_star+MAV.C_p_delta_r*delta_r_star) \
        + MAV.rho*MAV.S_wing*MAV.b*MAV.C_p_beta*np.sqrt(u_star**2+w_star**2)/(2.)
    Lp = MAV.gamma1*q_star + MAV.rho*Va_star*MAV.S_wing*MAV.b**2*MAV.C_p_p/4.
    Lr = -MAV.gamma2*q_star + MAV.rho*Va_star*MAV.S_wing*MAV.b**2*MAV.C_p_r/4.
    L_delta_a = MAV.rho*Va_star**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a/2.
    L_delta_r = MAV.rho*Va_star**2*MAV.S_wing*MAV.b*MAV.C_p_delta_r/2.
    Nv = MAV.rho*MAV.S_wing*MAV.b**2*v_star*(MAV.C_r_p*p_star+MAV.C_r_r*r_star)/(4.*Va_star) \
        + MAV.rho*MAV.S_wing*MAV.b*v_star*(MAV.C_r_0+MAV.C_r_beta*beta_star+MAV.C_r_delta_a*delta_a_star+MAV.C_r_delta_r*delta_r_star) \
        + MAV.rho*MAV.S_wing*MAV.b*MAV.C_r_beta*np.sqrt(u_star**2+w_star**2)/(2.)
    Np = MAV.gamma7*q_star + MAV.rho*Va_star*MAV.S_wing*MAV.b**2*MAV.C_r_p/4.
    Nr = -MAV.gamma1*q_star + MAV.rho*Va_star*MAV.S_wing*MAV.b**2*MAV.C_r_r/4.
    N_delta_a = MAV.rho*Va_star**2*MAV.S_wing*MAV.b*MAV.C_r_delta_a/2.
    N_delta_r = MAV.rho*Va_star**2*MAV.S_wing*MAV.b*MAV.C_r_delta_r/2.


    A_lat = np.array([[Yv, Yp/(Va_star*np.cos(beta_star)), Yr/(Va_star*np.cos(beta_star)), MAV.gravity*np.cos(theta_star)*np.cos(phi_star)/(Va_star*np.cos(beta_star)),0.0],
                      [Lv*Va_star*np.cos(beta_star), Lp, Lr, 0.0, 0.0],
                      [Nv*Va_star*np.cos(beta_star),Np,Nr,0.0,0.0],
                      [0.0, 1.0, np.cos(phi_star)*np.tan(theta_star),q_star*np.cos(phi_star)*np.tan(theta_star)-r_star*np.sin(phi_star)*np.tan(theta_star),0.0],
                      [0.0, 0.0, np.cos(phi_star)/np.cos(theta_star),p_star*np.cos(phi_star)/np.cos(theta_star)-r_star*np.sin(phi_star)/np.cos(theta_star),0.0]])
    print("A_lat",A_lat)
    B_lat = np.array([[Y_delta_a/(Va_star*np.cos(beta_star)),Y_delta_r/(Va_star*np.cos(beta_star))],
                      [L_delta_a,L_delta_r],
                      [N_delta_a,N_delta_r],
                      [0.0, 0.0],
                      [0.0, 0.0]])
    print("B_lat",B_lat)

    #lat_eig = np.linalg.eig(A_lat)
    #print(lat_eig)

    return A_lon, B_lon, A_lat, B_lat

def CD(alpha):
    '''
    UAV book equation 4.11
    '''
    result = MAV.C_D_p + ((MAV.C_L_0+alpha*MAV.C_L_alpha)**2)/(np.pi*MAV.e*MAV.AR)
    #result = (1-calcSigma(alpha))*(MAV.C_D_0 + MAV.C_D_alpha*alpha)+calcSigma(alpha)*(2.0*np.sign(alpha)*sin(alpha)**2*cos(alpha))
    return result

def CL(alpha):
    '''
    This is a linear coefficient model that is not valid over a wide
    range of angles of attack. UAV Book equation 4.13
    '''
    result = (1-calcSigma(alpha))*(MAV.C_L_0 + MAV.C_L_alpha*alpha)+calcSigma(alpha)*(2.0*np.sign(alpha)*np.sin(alpha)**2*np.cos(alpha))
    return result

def calcSigma(alpha):
    # blending function according to ch 4 UAV book slides
    nom = 1.0 + np.exp(-MAV.M*(alpha-MAV.alpha0))+np.exp(MAV.M*(alpha+MAV.alpha0))
    den = (1.0 + np.exp(-MAV.M*(alpha-MAV.alpha0)))*(1+np.exp(MAV.M*(alpha+MAV.alpha0)))
    result = nom/den
    return result

def euler_state(x_quat):
    # convert state x with attitude represented by quaternion
    # to x_euler with attitude represented by Euler angles
    x_euler = np.zeros((12,1))
    phi,theta,psi = Quaternion2Euler(x_quat[6:10])
    x_euler[0:6] = x_quat[0:6]
    x_euler[6] = phi
    x_euler[7] = theta
    x_euler[8] = psi
    x_euler[9:12] = x_quat[10:13]
    return x_euler

def quaternion_state(x_euler):
    # convert state x_euler with attitude represented by Euler angles
    # to x_quat with attitude represented by quaternions
    phi = x_euler[6]
    theta = x_euler[7]
    psi = x_euler[8]
    e0, e1, e2, e3 = Euler2Quaternion(phi,theta,psi)
    x_quat = x_euler.copy()
    x_quat[6] = e0
    x_quat[7] = e1
    x_quat[8] = e2
    x_quat = np.insert(x_quat,9,[e3],axis=0)
    return x_quat

def f_euler(mav, x_euler, input):
    # return 12x1 dynamics (as if state were Euler state)
    # compute f at euler_state
    return f_euler_

def df_dx(mav, x_euler, input):
    # take partial of f_euler with respect to x_euler
    return A

def df_du(mav, x_euler, delta):
    # take partial of f_euler with respect to delta
    return B

def dT_dVa(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to Va
    return dThrust

def dT_ddelta_t(mav, Va, delta_t):
    # returns the derivative of motor thrust with respect to delta_t
    return dThrust
