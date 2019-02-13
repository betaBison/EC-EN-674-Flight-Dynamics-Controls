from mav_dynamics import mav_dynamics
import numpy as np
import matplotlib.pyplot as plt
import parameters.simulation_parameters as SIM

mav = mav_dynamics(SIM.ts_simulation)


mav._state = np.array([[0.0],  # (0)
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

mav._update_velocity_data()
delta_a = 0.0
delta_e = 0.0
delta_r = 0.0
delta_t = 0.5
delta = np.array([[delta_a, delta_e, delta_r, delta_t]]).T
#current_wind = np.zeros((6,1))
forces_moments = mav._forces_moments(delta)
print(mav._derivatives(mav._state, forces_moments))

#mav.update_state(delta, current_wind)
#print(mav._state)
