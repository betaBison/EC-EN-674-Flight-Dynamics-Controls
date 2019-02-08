import sys
sys.path.append('..')
import parameters.aerosonde_parameters as MAV
from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np

def propThrust(delta_t,V_a):

    Omega_op = propOperatingSpeed(delta_t,V_a)

    J_op = 2.*pi*V_a/(Omega_op*MAV.D_prop)
    Ct_op = MAV.C_T2*J_op**2 + MAV.C_T1*J_op + MAV.C_T0
    print("Ct_op",Ct_op)

    result = Ct_op*MAV.rho*(Omega_op**2)*(MAV.D_prop**4)/((2.*pi)**2)
    return result

def propOperatingSpeed(delta_t,V_a):
    a = MAV.rho*MAV.D_prop**5*MAV.C_Q0/(2.*pi)**2
    b = MAV.rho*MAV.D_prop**4*MAV.C_Q1*V_a/(2.*pi) + MAV.KQ*MAV.K_V/MAV.R_motor
    c = MAV.rho*MAV.D_prop**3*MAV.C_Q2*V_a**2 - MAV.KQ*MAV.V_max*delta_t/MAV.R_motor + MAV.KQ*MAV.i0
    result = (-b + sqrt(b**2 - 4.*a*c))/(2.*a)
    return result

def propTorque(delta_t,V_a):
    Vin = MAV.V_max*delta_t
    Omega_op = propOperatingSpeed(delta_t,V_a)
    result = MAV.KQ*((Vin - MAV.K_V*Omega_op)/MAV.R_motor - MAV.i0)
    return result


airspeed = np.linspace(0,30,31)
voltages = np.linspace(0.2,1.0,5)
thrust = np.zeros((airspeed.shape[0],voltages.shape[0]))
torque = np.zeros((airspeed.shape[0],voltages.shape[0]))

for ii in range(airspeed.shape[0]):
    V_a = airspeed[ii]
    for jj in range(voltages.shape[0]):
        voltage = voltages[jj]
        thrust[ii,jj] = propThrust(voltage,V_a)
        torque[ii,jj] = propTorque(voltage,V_a)

plt.figure()
plt.plot(airspeed,thrust)
plt.xlim([0,30])
plt.ylim([-5,90])

plt.figure()
plt.plot(airspeed,torque)
plt.xlim([0,30])
plt.ylim([-0.5,3.0])

plt.show()
