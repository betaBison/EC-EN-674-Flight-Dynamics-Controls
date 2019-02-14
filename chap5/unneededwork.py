



def trim_objective(x,mav,Va,gamma)
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
    Cx = -CD(alpha_star)*np.cos(alpha_star) + CL(alpha_star)*np.sin(alpha_star)
    Cxq = -MAV.C_D_q*np.cos(alpha_star) + MAV.C_L_q*np.sin(alpha_star)
    Cxde = -MAV.C_D_delta_e*np.cos(alpha_star) + MAV.C_L_delta_e*np.sin(alpha_star)

    delta_t_star = np.sqrt((2.*MAV.mass*(-r_star*v_star+q_star*w_star+MAV.gravity*np.sin(theta_star)) \
        -MAV.rho*Va_star**2*MAV.S_wing*(Cx+Cxq*MAV.c*q_star/(2.*Va_star)+Cxde*delta_e_star)) \
        /(MAV.rho*MAV.S_prop*MAV.C_prop*MAV.k_motor**2)+Va_star**2/MAV.k_motor**2)

    # F.3
    A = np.array([[MAV.C_p_delta_a,MAV.C_p_delta_r],
                  [MAV.C_r_delta_a,MAV.C_r_delta_r]])
    B = np.array([[(-MAV.gamma1*p_star*q_star+MAV.gamma2*q_star*r_star)/(0.5*MAV.rho*Va_star**2*MAV.S_wing*MAV.b)
                    -MAV.C_p_0-MAV.C_p_beta*beta_star-MAV.C_p_p*MAV.b*p_star/(2.*Va_star)-MAV.C_p_r*MAV.b*r_star/(2.*Va_star)],
                   [(-MAV.gamma7*p_star*q_star+MAV.gamma1*q_star*r_star)/(0.5*MAV.rho*Va_star**2*MAV.S_wing*MAV.b)
                    -MAV.C_r_0-MAV.C_r_beta*beta_star-MAV.C_r_p*MAV.b*p_star/(2.*Va_star)-MAV.C_r_r*MAV.b*r_star/(2.*Va_star)]])
    delta_result = np.matmul(np.linalg.inv(A),B)
    delta_a_star = delta_result.item(0)
    delta_r_star = delta_result.item(1)

    delta = np.array([[delta_a_star, delta_e_star, delta_r_star, delta_t_star]]).T

    f_result = calcLinearDerivatives(x_dot_star,delta,alpha_star,Va_star,beta_star)



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
    Cx = -CD(alpha)*np.cos(alpha) + CL(alpha)*np.sin(alpha)
    Cxq = -MAV.C_D_q*np.cos(alpha) + MAV.C_L_q*np.sin(alpha)
    Cxde = -MAV.C_D_delta_e*np.cos(alpha) + MAV.C_L_delta_e*np.sin(alpha)
    Cz = -CD(alpha)*np.sin(alpha) - CL(alpha)*np.cos(alpha)
    Czq = -MAV.C_D_q*np.sin(alpha) - MAV.C_L_q*np.cos(alpha)
    Czde = -MAV.C_D_delta_e*np.sin(alpha) - MAV.C_L_delta_e*np.cos(alpha)

    u_dot = r*v-q*w + MAV.gravity*np.sin(theta) + MAV.rho*Va**2*MAV.S_wing*(Cx+Cxq*MAV.c*q/(2.*Va)+Cxde*delta_e)/(2.*MAV.mass) + \
        MAV.rho*MAV.S_prop*MAV.C_prop*((MAV.k_motor*delta_t)**2-Va*2)/(2.*MAV.mass)
    v_dot = p*w-r*u + MAV.gravity*np.cos(theta)*np.sin(phi)+ \
        MAV.rho*Va**2*MAV.S_wing*(MAV.C_Y_0+MAV.C_Y_beta*beta*MAV.C_Y_p*MAV.b*p/(2.*Va)+MAV.C_Y_r*MAV.b*r/(2.*Va)+MAV.C_Y_delta_a*delta_a+MAV.C_Y_delta_r*delta_r)/(2.*MAV.mass)
    w_dot = q*u-p*v + MAV.gravity*np.cos(theta)*np.cos(phi)+MAV.rho*Va**2*MAV.S_wing*(Cz+Czq*MAV.c*q/(2.*Va)+Czde*delta_e)/(2.*MAV.mass)

    # rotational kinematics
    phi_dot = p + q*np.sin(phi)*np.tan(theta)+r*np.cos(phi)*np.tan(theta)
    theta_dot = q*np.cos(phi)-r*np.sin(phi)
    psi_dot = q*np.sin(phi)/np.cos(theta)+r*np.cos(phi)/np.cos(theta)

    # rotatonal dynamics
    # check the multiply that should be minus p_dot = MAV.gamma1*p*q - MAV.gamma2*q*r + \
        0.5*MAV.rho*Va**2*MAV.S_wing*MAV.b*(MAV.C_p_0+MAV.C_p_beta*beta-MAV.C_p_p*MAV.b*p/(2.*Va)+MAV.C_p_r*MAV.b*r/(2.*Va)+MAV.C_p_delta_a*delta_a+MAV.C_p_delta_r*delta_r)
    q_dot = MAV.gamma5*p*r - MAV.gamma6*(p**2-r**2) + MAV.rho*Va**2*MAV.S_wing*MAV.c*(MAV.C_m_0+MAV.C_m_alpha*alpha+MAV.C_m_q*MAV.c*q/(2.*Va)+MAV.C_m_delta_e*delta_e)/(2.*MAV.Jy)
    # check the multiply that should be minus r_dot = MAV.gamma7*p*q - MAV.gamma1*q*r + \
        0.5*MAV.rho*Va**2*MAV.S_wing*MAV.b*(MAV.C_r_0+MAV.C_r_beta*beta*MAV.C_r_p*MAV.b*p/(2.*Va)+MAV.C_r_r*MAV.b*r/(2.*Va)+MAV.C_r_delta_a*delta_a+MAV.C_r_delta_r*delta_r)

    # collect the derivative of the states
    x_dot = np.array([[pn_dot, pe_dot, pd_dot, u_dot, v_dot, w_dot,
                       phi_dot, theta_dot, psi_dot, p_dot, q_dot, r_dot]]).T
    return x_dot



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
