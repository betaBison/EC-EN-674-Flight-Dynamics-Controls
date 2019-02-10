from mav_dynamics import mav_dynamics
import numpy as np
import matplotlib.pyplot as plt

mav = mav_dynamics(0.01)

fig = plt.figure(figsize=(100,100))
ax21 = fig.add_subplot(6,4,21)
ax22 = fig.add_subplot(6,4,22,sharey=ax21)
ax23 = fig.add_subplot(6,4,23,sharey=ax21)
ax24 = fig.add_subplot(6,4,24,sharey=ax21)

ax1 = fig.add_subplot(6,4,1,sharex=ax21)
ax2 = fig.add_subplot(6,4,2,sharey=ax1,sharex=ax22)
ax3 = fig.add_subplot(6,4,3,sharey=ax1,sharex=ax23)
ax4 = fig.add_subplot(6,4,4,sharey=ax1,sharex=ax24)

ax5 = fig.add_subplot(6,4,5,sharex=ax21)
ax6 = fig.add_subplot(6,4,6,sharey=ax5,sharex=ax22)
ax7 = fig.add_subplot(6,4,7,sharey=ax5,sharex=ax23)
ax8 = fig.add_subplot(6,4,8,sharey=ax5,sharex=ax24)

ax9 = fig.add_subplot(6,4,9,sharex=ax21)
ax10 = fig.add_subplot(6,4,10,sharey=ax9,sharex=ax22)
ax11 = fig.add_subplot(6,4,11,sharey=ax9,sharex=ax23)
ax12 = fig.add_subplot(6,4,12,sharey=ax9,sharex=ax24)

ax13 = fig.add_subplot(6,4,13,sharex=ax21)
ax14 = fig.add_subplot(6,4,14,sharey=ax13,sharex=ax22)
ax15 = fig.add_subplot(6,4,15,sharey=ax13,sharex=ax23)
ax16 = fig.add_subplot(6,4,16,sharey=ax13,sharex=ax24)

ax17 = fig.add_subplot(6,4,17,sharex=ax21)
ax18 = fig.add_subplot(6,4,18,sharey=ax17,sharex=ax22)
ax19 = fig.add_subplot(6,4,19,sharey=ax17,sharex=ax23)
ax20 = fig.add_subplot(6,4,20,sharey=ax17,sharex=ax24)


delta_e = np.linspace(-0.3,0.3,num=10)
delta_t = 0.8
delta_a = 0.0
delta_r = 0.0
fm = np.zeros((6,np.shape(delta_e)[0]))
for i in range(np.shape(fm)[1]):
  delta = np.array([[delta_a,delta_e.item(i),delta_r,delta_t]])
  fm[:,i] = mav._forces_moments(delta).reshape(6,)


ylabels = ['Fx','Fy','Fz','Mx','My','Mz']
axs = [ax1, ax5, ax9, ax13, ax17, ax21]
for i in range(6):
  ax = axs[i]
  ax.plot(delta_e,fm[i,:],lw=2.5)
  ax.set_ylabel(ylabels[i])
  ax.grid(True)
  if i != 5:
    [label.set_visible(False) for label in ax.get_xticklabels()]

ax.set_xlabel('delta_e')
ax.set_xlim((delta_e[0],delta_e[-1]))


delta_e = 0.
delta_t = np.linspace(0.4,1.0,num=10)
fm = np.zeros((6,np.shape(delta_t)[0]))
for i in range(np.shape(fm)[1]):
  delta = np.array([[delta_a,delta_e,delta_r,delta_t.item(i)]])
  fm[:,i] = mav._forces_moments(delta).reshape(6,)

axs = [ax2,ax6,ax10,ax14,ax18,ax22]
for i in range(6):
  ax = axs[i]
  ax.plot(delta_t,fm[i,:],lw=2.5)
  ax.grid(True)
  [label.set_visible(False) for label in ax.get_yticklabels()]
  if i != 5:
    [label.set_visible(False) for label in ax.get_xticklabels()]

ax.set_xlabel('delta_t')
ax.set_xlim((delta_t[0],delta_t[-1]))




delta_t = 0.8
delta_a = np.linspace(-0.2,0.2,num=10)
fm = np.zeros((6,np.shape(delta_a)[0]))
for i in range(np.shape(fm)[1]):
  delta = np.array([[delta_a.item(i),delta_e,delta_r,delta_t,]])
  fm[:,i] = mav._forces_moments(delta).reshape(6,)


axs = [ax3,ax7,ax11,ax15,ax19,ax23]
for i in range(6):
  ax = axs[i]
  ax.plot(delta_a,fm[i,:],lw=2.5)
  ax.grid(True)
  [label.set_visible(False) for label in ax.get_yticklabels()]
  if i != 5:
    [label.set_visible(False) for label in ax.get_xticklabels()]

ax.set_xlabel('delta_a')
ax.set_xlim((delta_a[0],delta_a[-1]))



delta_a = 0.0
delta_r = np.linspace(-0.02,0.02,num=10)
fm = np.zeros((6,np.shape(delta_r)[0]))
for i in range(np.shape(fm)[1]):
  delta = np.array([[delta_a,delta_e,delta_r.item(i),delta_t]])
  fm[:,i] = mav._forces_moments(delta).reshape(6,)


axs = [ax4,ax8,ax12,ax16,ax20,ax24]
for i in range(6):
  ax = axs[i]
  ax.plot(delta_r,fm[i,:],lw=2.5)
  ax.grid(True)
  [label.set_visible(False) for label in ax.get_yticklabels()]
  if i != 5:
    [label.set_visible(False) for label in ax.get_xticklabels()]
ax.set_xlabel('delta_r')
ax.set_xlim((delta_r[0],delta_r[-1]))





plt.show()
