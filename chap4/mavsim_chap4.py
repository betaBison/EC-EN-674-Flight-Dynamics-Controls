"""
mavsimPy
    - Chapter 4 assignment for Beard & McLain, PUP, 2012
    - Update history:
        12/27/2018 - RWB
        1/17/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

from chap2.mav_viewer import mav_viewer
#from chap2.video_writer import video_writer
from chap3.data_viewer import data_viewer
from chap4.mav_dynamics import mav_dynamics
#from chap4.wind_simulation import wind_simulation

# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if VIDEO == True:
    video = video_writer(video_name="chap4_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
#wind = wind_simulation(SIM.ts_simulation)
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
switch = 0
while sim_time < SIM.end_time:
    if switch == 0:
        if sim_time < SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = 0.01 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 1:
        if sim_time < 2*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = -0.005 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 2:
        if sim_time <3*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = 0.005 # about trim
            delta_e = 0.05 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 3:
        if sim_time < 4*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = -0.005 #0.018 # about trim
            delta_e = -0.4 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 4:
        if sim_time < 5*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = 0.005 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.008 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 5:
        if sim_time < 6*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = 0.005 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.000 #0.005
            delta_t = 0.5
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 6:
        if sim_time < 7*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = -0.0155 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.9
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1
    elif switch == 7:
        if sim_time < 8*SIM.end_time/8.:
            #-------set control surfaces-------------
            delta_a = 0.015 #0.018 # about trim
            delta_e = -0.2 #0.9#-0.2
            delta_r = 0.00 #0.005
            delta_t = 0.1
        else:
            mav.__init__(SIM.ts_simulation)
            switch += 1

    delta = np.array([[delta_a, delta_e, delta_r, delta_t]]).T  # transpose to make it a column vector

    #-------physical system-------------
    #current_wind = wind.update()  # get the new wind vector
    current_wind = np.zeros((6,1))
    mav.update_state(delta, current_wind)  # propagate the MAV dynamics


    #-------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV

    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)

    if VIDEO == True:
        video.update(sim_time)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True:
    video.close()
