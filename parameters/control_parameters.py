import sys
sys.path.append('..')
import numpy as np
#import chap5.transfer_function_coef as TF
import parameters.aerosonde_parameters as MAV
import chap5.trim_results as TR

gravity = MAV.gravity
sigma =
Va0 =

#----------roll loop-------------
# design parameters
roll_wn = 10.
roll_zeta = 0.707
# calculations
a_phi_1 = -0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_p*MAV.b/(2.*mav._Va)
a_phi_2 = 0.5*MAV.rho*mav._Va**2*MAV.S_wing*MAV.b*MAV.C_p_delta_a

roll_kp = roll_wn**2/a_phi_2
roll_kd = (2.*roll_zeta*roll_wn - a_phi_1)/a_phi_2

#----------course loop-------------
# design parameters
coure_wn = 10.
course_zeta = 0.707
# calculations
course_kp = 2.*course_zeta*course_wn*
course_ki =

#----------sideslip loop-------------
sideslip_ki =
sideslip_kp =

#----------yaw damper-------------
yaw_damper_tau_r = 0.1
yaw_damper_kp = 0.5

#----------pitch loop-------------
pitch_kp =
pitch_kd =
K_theta_DC =

#----------altitude loop-------------
altitude_kp =
altitude_ki =
altitude_zone =

#---------airspeed hold using throttle---------------
airspeed_throttle_kp =
airspeed_throttle_ki =
