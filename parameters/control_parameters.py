import sys
sys.path.append('..')
import numpy as np
#import chap5.transfer_function_coef as TF
import parameters.aerosonde_parameters as MAV
import chap5.trim_results as TR

gravity = MAV.gravity
#sigma =
Va0 = TR.Va
Vg = TR.Vg
alpha = TR.alpha
delta_e = TR.trim_input.item(1)
delta_t = TR.trim_input.item(3)



#----------roll loop-------------
# design parameters
roll_wn = 8.
roll_zeta = 1.707
# calculations
a_phi_1 = -0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2.*Va0)
a_phi_2 = 0.5*MAV.rho*Va0**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a
roll_kp = roll_wn**2/a_phi_2
roll_kd = (2.*roll_zeta*roll_wn - a_phi_1)/a_phi_2

#----------course loop-------------
# design parameters
W_roll_course_separation = 7.
course_zeta = 1.1

# calculations
course_wn = roll_wn/W_roll_course_separation
course_kp = 2.*course_zeta*course_wn*Vg/gravity
course_ki = 0.1*(course_wn**2*Vg/gravity)

#----------sideslip loop-------------
#sideslip_ki =
#sideslip_kp =

#----------yaw damper-------------
# design parameters
yaw_damper_tau_r = 0.1
yaw_damper_kp = 0.5

#----------pitch loop-------------
# design parameters
pitch_wn = 13.
pitch_zeta = 0.6
# calculations
a_theta_1 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing*MAV.C_m_q*MAV.c/(2.*MAV.Jy*2.*Va0)
a_theta_2 = -MAV.rho*Va0**2*MAV.c*MAV.S_wing*MAV.C_m_alpha/(2.*MAV.Jy)
a_theta_3 = MAV.rho*Va0**2*MAV.c*MAV.S_wing*MAV.C_m_delta_e/(2.*MAV.Jy)
pitch_kp = (pitch_wn**2 - a_theta_2)/a_theta_3
pitch_kd = (2.*pitch_zeta*pitch_wn-a_theta_1)/a_theta_3
K_theta_DC = (pitch_kp*a_theta_3)/pitch_wn**2

#----------altitude loop-------------
# design Parameters
W_pitch_altitude_separation = 35.
altitude_zeta = 0.707
#calculations
altitude_wn = pitch_wn/W_pitch_altitude_separation
altitude_kp = (2.*altitude_zeta*altitude_wn)/(K_theta_DC*Va0)
altitude_ki = altitude_wn**2/(K_theta_DC*Va0)

#altitude_zone =

#---------airspeed hold using throttle---------------
# design parameters
airspeed_throttle_wn = 2.
airspeed_throttle_zeta = 2.0
# calcualtions
a_v_1 = MAV.rho*Va0*MAV.S_wing*(MAV.C_D_0 + MAV.C_D_alpha*alpha+MAV.C_D_delta_e*delta_e)/MAV.mass + MAV.rho*MAV.S_prop*MAV.C_prop*Va0/MAV.mass
a_v_2 = MAV.rho*MAV.S_prop*MAV.C_prop*MAV.k_motor**2*delta_t/MAV.mass
airspeed_throttle_kp = (2.*airspeed_throttle_zeta*airspeed_throttle_wn - a_v_1)/a_v_2
airspeed_throttle_ki = airspeed_throttle_wn**2/a_v_2
