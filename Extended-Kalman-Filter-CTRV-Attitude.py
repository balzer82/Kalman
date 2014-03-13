# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, latex, Plot
from sympy.interactive import printing
printing.init_printing()
%pylab inline --no-import-all

# <headingcell level=1>

# Adaptive Extended Kalman Filter Implementation for Constant Turn Rate and Velocity (CTRV) Vehicle Model with Attitude Estimation in Python

# <markdowncell>

# Situation covered: You have an velocity sensor which measures the vehicle speed ($v$) in heading direction ($\psi$) and a yaw rate sensor ($\dot \psi$) which both have to fused with the position ($x$ & $y$) from a GPS sensor.

# <headingcell level=2>

# State Vector - Constant Turn Rate and Velocity Vehicle Model (CTRV) + Roll and Pitch Estimation

# <markdowncell>

# $$x_k= \left[\begin{matrix}x\\y\\\psi\\v\\\dot\psi\\\phi\\\dot\phi\\\Theta\\\dot\Theta\end{matrix}\right] = \left[ \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Heading} \\ \text{Velocity} \\ \text{Yaw Rate} \\ \text{Pitch} \\ \text{Pitchrate} \\ \text{Roll} \\ \text{Rollrate}} \right]$$

# <codecell>

numstates=9 # States

# <codecell>

dt = 1.0/50.0 # Sample Rate of the Measurements is 50Hz
dtGPS=1.0/10.0 # Sample Rate of GPS is 10Hz

# <markdowncell>

# All symbolic calculations are made with [Sympy](http://nbviewer.ipython.org/github/jrjohansson/scientific-python-lectures/blob/master/Lecture-5-Sympy.ipynb). Thanks!

# <codecell>

vs, psis, dpsis, dts, xs, ys, phis, dphis, thetas, dthetas = \
 symbols('v \psi \dot\psi T x y \phi \dot\phi \Theta \dot\Theta')

As = Matrix([[xs+(vs/dpsis)*(sin(psis+dpsis*dts)-sin(psis))],
             [ys+(vs/dpsis)*(-cos(psis+dpsis*dts)+cos(psis))],
             [psis+dpsis*dts],
             [vs],
             [dpsis],
             [phis+dphis*dts],
             [dphis],
             [thetas+dthetas*dts],
             [dthetas]])
state = Matrix([xs,ys,psis,vs,dpsis,phis,dphis,thetas,dthetas])

# <headingcell level=2>

# Initial Uncertainty

# <markdowncell>

# Initialized with $0$ means you are pretty sure where the vehicle starts

# <codecell>

P = 1000.0*np.eye(numstates)
print(P.shape)

fig = plt.figure(figsize=(numstates/2, numstates/2))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Initial Covariance Matrix $P$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(10))
# set the locations and labels of the yticks
plt.yticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(10))
# set the locations and labels of the yticks
plt.xticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,8.5])
plt.ylim([8.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <headingcell level=2>

# Dynamic Matrix

# <markdowncell>

# This formulas calculate how the state is evolving from one to the next time step

# <codecell>

As

# <headingcell level=3>

# Calculate the Jacobian of the Dynamic Matrix with respect to the state vector

# <codecell>

state

# <codecell>

As.jacobian(state)

# <markdowncell>

# It has to be computed on every filter step because it consists of state variables.

# <headingcell level=2>

# Control Input

# <markdowncell>

# Matrix G is the Jacobian of the Dynamic Matrix with respect to control (the translation velocity $v$ and the rotational
# velocity $\dot \psi$).

# <codecell>

control = Matrix([vs,dpsis,dphis,dthetas])
control

# <headingcell level=3>

# Calculate the Jacobian of the Dynamic Matrix with Respect to the Control

# <codecell>

Gs=As.jacobian(control)
Gs

# <markdowncell>

# It has to be computed on every filter step because it consists of state variables.

# <headingcell level=2>

# Process Noise Covariance Matrix $Q$

# <markdowncell>

# Matrix Q is the expected noise on the State.
# 
# One method is based on the interpretation of the matrix as the weight of the dynamics prediction from the state equations
# Q relative to the measurements.
# 
# As you can see in [Schubert, R., Adam, C., Obst, M., Mattern, N., Leonhardt, V., & Wanielik, G. (2011). Empirical evaluation of vehicular models for ego motion estimation. 2011 IEEE Intelligent Vehicles Symposium (IV), 534–539. doi:10.1109/IVS.2011.5940526] one can assume the velocity process noise for a vehicle with $\sigma_v=1.5m/s$ and the yaw rate process noise with $\sigma_\psi=0.29rad/s$, when a timestep takes 0.02s (50Hz).

# <codecell>

control

# <codecell>

svQ = (1.5)**2      # Velocity
syQ = (0.29*dt)**2  # Yawrate
spQ = (0.2*dt)**2   # Pitchrate
srQ = (0.8*dt)**2   # Rollrate

Q = np.matrix([[svQ, 0.0, 0.0, 0.0],
               [0.0, syQ, 0.0, 0.0],
               [0.0, 0.0, spQ, 0.0],
               [0.0, 0.0, 0.0, srQ]])

# <codecell>

fig = plt.figure(figsize=(4, 4))
im = plt.imshow(Q, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Process Noise Covariance Matrix $Q$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(5))
# set the locations and labels of the yticks
plt.yticks(np.arange(4), \
           ('$v$', '$\dot \psi$', '$\dot \phi$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(5))
# set the locations and labels of the yticks
plt.xticks(np.arange(4), \
           ('$v$', '$\dot \psi$', '$\dot \phi$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <headingcell level=2>

# Real Measurements

# <codecell>

path = './../RaspberryPi-CarPC/TinkerDataLogger/DataLogs/2014/'
datafile = path+'2014-02-21-002-Data.csv'

date, \
time, \
millis, \
ax, \
ay, \
az, \
rollrate, \
pitchrate, \
yawrate, \
roll, \
pitch, \
yaw, \
speed, \
course, \
latitude, \
longitude, \
altitude, \
pdop, \
hdop, \
vdop, \
epe, \
fix, \
satellites_view, \
satellites_used, \
temp = np.loadtxt(datafile, delimiter=',', unpack=True, skiprows=1000)

print('Read \'%s\' successfully.' % datafile)

# A course of 0° means the Car is traveling north bound
# and 90° means it is traveling east bound.
# In the Calculation following, East is Zero and North is 90°
# We need an offset.
course =(-course+90.0)

# <headingcell level=3>

# Static Gain

# <codecell>

pitchrate = pitchrate - 0.683613
rollrate = rollrate - 0.433898

# <codecell>

# clamp speed and yawrate to zero while standing still
speed[speed<5.0]=0.0
yawrate[speed<5.0]=0.0

# <headingcell level=2>

# Measurement Noise Covariance Matrix $R$ (Adaptive)

# <markdowncell>

# "In practical use, the uncertainty estimates take on the significance of relative weights of state estimates and measurements. So it is not so much important that uncertainty is absolutely correct as it is that it be relatively consistent across all models" - Kelly, A. (1994). A 3D state space formulation of a navigation Kalman filter for autonomous vehicles, (May). Retrieved from http://oai.dtic.mil/oai/oai?verb=getRecord&metadataPrefix=html&identifier=ADA282853

# <markdowncell>

# Because the estimation of Roll and Pitch is only valid for quasistatic situations (which is not valid for a moving vehicle), the values for the measured rotation $\sigma_r$ is very high.

# <codecell>

sp = 6.0**2
sr = (100.0*dt)**2
R = np.matrix([[sp, 0.0, 0.0, 0.0],
               [0.0, sp, 0.0, 0.0],
               [0.0, 0.0, sr, 0.0],
               [0.0, 0.0, 0.0, sr]])

# <codecell>

fig = plt.figure(figsize=(4, 4))
im = plt.imshow(R, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Measurement Noise Covariance Matrix $R$')
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(5))
# set the locations and labels of the yticks
plt.yticks(np.arange(4), \
           ('$x$', '$y$', '$\phi$', '$\Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(5))
# set the locations and labels of the yticks
plt.xticks(np.arange(4), \
           ('$x$', '$y$', '$\phi$', '$\Theta$'),\
           fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <markdowncell>

# R is here just initialized. In the Kalman Filter Step it will calculated dynamically with the $EPE$ (Estimated Position Error) from the GPS signal as well as depending on the $speed$, like proposed in [Wender, S. (2008). Multisensorsystem zur erweiterten Fahrzeugumfelderfassung. Retrieved from http://vts.uni-ulm.de/docs/2008/6605/vts_6605_9026.pdf P.108].

# <markdowncell>

# $\sigma_p^2 = \sigma_\text{speed}^2 + \sigma_\text{EPE}^2$
# 
# $\sigma_v = (v+0.01)^{-1}$
# 
# $\sigma_\text{EPE} = 10 \cdot EPE$

# <codecell>

plt.figure(figsize=(16,6))
spspeed = (1.0/((speed/3.6)+0.01))
spepe = (10.0)*epe
sp = spspeed**2 + spepe**2
plt.semilogy(spspeed, label='$\sigma_P$ from speed')
plt.semilogy(spepe, label='$\sigma_P$ from EPE')
plt.semilogy(sp, label='Resulting $R$ value')
plt.ylabel('Values for $R$ Matrix')
plt.xlabel('Filterstep ($k$)')
plt.legend()
plt.savefig('Extended-Kalman-Filter-CTRV-Adaptive-R.png', dpi=72, transparent=True, bbox_inches='tight')

# <codecell>


# <headingcell level=2>

# Measurement Function H

# <markdowncell>

# Matrix H is the Jacobian of the Measurement function h with respect to the state.

# <codecell>

hs = Matrix([[xs],
             [ys],
             [phis],
             [thetas]])
Hs=hs.jacobian(state)
Hs

# <headingcell level=3>

# Identity Matrix

# <codecell>

I = np.eye(numstates)
print(I, I.shape)

# <headingcell level=3>

# Roll & Pitch from Acceleration

# <markdowncell>

# As mentioned in Euston, M., Coote, P., & Mahony, R. (2008). A complementary filter for attitude estimation of a fixed-wing UAV. Intelligent Robots and …, 340–345. doi:10.1109/IROS.2008.4650766, a raw estimate of roll and pitch can be determined from acceleration with respect to gravity.

# <codecell>

rollacc = np.arctan2(-ay, -az)
pitchacc= -np.arctan2(-ax, np.sqrt(ay**2+az**2))

# <headingcell level=3>

# Lat/Lon to Meters

# <codecell>

RadiusEarth = 6378388.0 # m
arc= 2.0*np.pi*RadiusEarth/360.0 # m/°

dx = arc * np.cos(latitude*np.pi/180.0) * np.hstack((0.0, np.diff(longitude))) # in m
dy = arc * np.hstack((0.0, np.diff(latitude))) # in m

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx**2+dy**2)

GPS=np.hstack((True, (np.diff(ds)>0.0).astype('bool'))) # GPS Trigger for Kalman Filter

# <headingcell level=2>

# Initial State

# <codecell>

x = np.matrix([[mx[0], my[0], course[0]/180.0*np.pi-0.3, speed[0]/3.6+0.001, yawrate[0]/180.0*np.pi, \
                0.0, pitchrate[0]/180.0*np.pi, \
                0.0, rollrate[0]/180.0*np.pi]]).T
print(x, x.shape)

U=float(np.cos(x[2])*x[3])
V=float(np.sin(x[2])*x[3])

plt.quiver(x[0], x[1], U, V)
plt.scatter(float(x[0]), float(x[1]), s=100)
plt.title('Initial Location')
plt.axis('equal')

# <headingcell level=3>

# Put everything together as a measurement vector

# <codecell>

measurements = np.vstack((mx, my, pitchacc, rollacc))
# Lenth of the measurement
m = measurements.shape[1]
print(measurements.shape)

# <codecell>


# <codecell>

# Preallocation for Plotting
x0 = []
x1 = []
x2 = []
x3 = []
x4 = []
x5 = []
x6 = []
x7 = []
x8 = []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
dstate=[]
K = np.matrix([[0.0], [0.0], [0.0], [0.0], [0.0]])
Z = np.matrix([[0.0],[0.0],[0.0]])

# <headingcell level=2>

# Extended Kalman Filter

# <markdowncell>

# ![Extended Kalman Filter Step](https://raw.github.com/balzer82/Kalman/master/Extended-Kalman-Filter-Step.png)

# <markdowncell>

# $$x_k= \left[\begin{matrix}x\\y\\\psi\\v\\\dot\psi\\\phi\\\dot\phi\\\Theta\\\dot\Theta\end{matrix}\right] = \left[ \matrix{ \text{Position X} \\ \text{Position Y} \\ \text{Heading} \\ \text{Velocity} \\ \text{Yaw Rate} \\ \text{Pitch} \\ \text{Pitchrate} \\ \text{Roll} \\ \text{Rollrate}} \right] =  \underbrace{\begin{matrix}x[0] \\ x[1] \\ x[2] \\ x[3] \\ x[4] \\ x[5] \\ x[6] \\ x[7] \\ x[8] \end{matrix}}_{\textrm{Python Nomenclature}}$$

# <codecell>

for filterstep in range(m):
    
    # Data (Control)
    vt=speed[filterstep]/3.6
    yat=yawrate[filterstep]/180.0*np.pi
    pit=pitchrate[filterstep]/180.0*np.pi
    rot=rollrate[filterstep]/180.0*np.pi
    
   
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"
    if np.abs(yat)<0.0001: # Driving straight
        x[0] = x[0] + vt*dt * np.cos(x[2])
        x[1] = x[1] + vt*dt * np.sin(x[2])
        x[2] = x[2]
        x[3] = vt
        x[4] = -0.0000001 # avoid numerical issues in Jacobians
        x[5] = x[5] + pit*dt
        x[6] = pit
        x[7] = x[7] + rot*dt
        x[8] = rot
        dstate.append(0)
    else: # otherwise
        x[0] = x[0] + (vt/yat) * (np.sin(yat*dt+x[2]) - np.sin(x[2]))
        x[1] = x[1] + (vt/yat) * (-np.cos(yat*dt+x[2])+ np.cos(x[2]))
        x[2] = (x[2] + yat*dt + np.pi) % (2.0*np.pi) - np.pi
        x[3] = vt
        x[4] = yat
        x[5] = x[5] + pit*dt
        x[6] = pit
        x[7] = x[7] + rot*dt
        x[8] = rot
        dstate.append(1)
    
    # Calculate the Jacobian of the Dynamic Matrix A
    # see "Calculate the Jacobian of the Dynamic Matrix with respect to the state vector"
    a13 = float((x[3]/x[4]) * (np.cos(x[4]*dt+x[2]) - np.cos(x[2])))
    a14 = float((1.0/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a15 = float((dt*x[3]/x[4])*np.cos(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a23 = float((x[3]/x[4]) * (np.sin(x[4]*dt+x[2]) - np.sin(x[2])))
    a24 = float((1.0/x[4]) * (-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    a25 = float((dt*x[3]/x[4])*np.sin(x[4]*dt+x[2]) - (x[3]/x[4]**2)*(-np.cos(x[4]*dt+x[2]) + np.cos(x[2])))
    JA = np.matrix([[1.0, 0.0, a13, a14, a15, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, a23, a24, a25, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0,  dt, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,  dt],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    # Calculate the Jacobian of the Control Input G
    # see "Calculate the Jacobian of the Dynamic Matrix with Respect to the Control"
    g11 = float(1.0/x[4]*(-np.sin(x[2])+np.sin(dt*x[4]+x[2])))
    g12 = float(dt*x[3]/x[4]*np.cos(dt*x[4]+x[2]) - x[3]/x[4]**2*(-np.sin(x[2])+np.sin(dt*x[4]+x[2])))
    g21 = float(1.0/x[4]*(np.cos(x[2])-np.cos(dt*x[4]+x[2])))
    g22 = float(dt*x[3]/x[4]*np.sin(dt*x[4]+x[2]) - x[3]/x[4]**2*(np.cos(x[2])-np.cos(dt*x[4]+x[2])))
    JG = np.matrix([[g11, g12, 0.0, 0.0],
                    [g21, g22, 0.0, 0.0],
                    [0.0, dt, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, dt, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, dt],
                    [0.0, 0.0, 0.0, 1.0]])
    
    # Project the error covariance ahead
    P = JA*P*JA.T + JG*Q*JG.T
    
    
    # Measurement Update (Correction)
    # ===============================

    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])],
                    [float(x[5])],
                    [float(x[7])]])
    
    # Because GPS is sampled with 10Hz and the other Measurements, as well as
    # the filter are sampled with 50Hz, one have to wait for correction until
    # there is a new GPS Measurement
    
    if GPS[filterstep]:
        # Calculate the Jacobian of the Measurement Function
        # see "Measurement Matrix H"
        H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    else:
        H = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])        
    
        
    # Calculate R with Data from the GPS Signal itself
    # and raise it when standing still
    R[0,0] = sp[filterstep]
    R[1,1] = sp[filterstep]
    
    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.inv(S)

    # Update the estimate via
    Z = measurements[:,filterstep].reshape(H.shape[0],1)
    y = Z - (hx)                         # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*H))*P


    # Save states for Plotting
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    x4.append(float(x[4]))
    x5.append(float(x[5]))
    x6.append(float(x[6]))
    x7.append(float(x[7]))
    x8.append(float(x[8]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))    
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Pddx.append(float(P[4,4]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    Kddx.append(float(K[4,0]))

# <headingcell level=2>

# Plots

# <codecell>

%pylab inline --no-import-all

# <headingcell level=3>

# Uncertainty

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.semilogy(range(m),Px, label='$x$')
plt.step(range(m),Py, label='$y$')
plt.step(range(m),Pdx, label='$\psi$')
plt.step(range(m),Pdy, label='$v$')
plt.step(range(m),Pddx, label='$\dot \psi$')

plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Uncertainty (Elements from Matrix $P$)')
plt.legend(loc='best',prop={'size':22})

# <codecell>

fig = plt.figure(figsize=(numstates, numstates))
im = plt.imshow(P, interpolation="none", cmap=plt.get_cmap('binary'))
plt.title('Covariance Matrix $P$ (after %i Filter Steps)' % (m))
ylocs, ylabels = plt.yticks()
# set the locations of the yticks
plt.yticks(np.arange(10))
# set the locations and labels of the yticks
plt.yticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

xlocs, xlabels = plt.xticks()
# set the locations of the yticks
plt.xticks(np.arange(10))
# set the locations and labels of the yticks
plt.xticks(np.arange(9), \
           ('$x$', '$y$', '$\psi$', '$v$', '$\dot \psi$', '$\phi$', '$\dot \phi$', '$\Theta$', '$\dot \Theta$'),\
           fontsize=22)

plt.xlim([-0.5,8.5])
plt.ylim([8.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)

plt.tight_layout()

# <headingcell level=3>

# Kalman Gains

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.step(range(len(measurements[0])),Kx, label='$x$')
plt.step(range(len(measurements[0])),Ky, label='$y$')
plt.step(range(len(measurements[0])),Kdx, label='$\psi$')
plt.step(range(len(measurements[0])),Kdy, label='$v$')
plt.step(range(len(measurements[0])),Kddx, label='$\dot \psi$')


plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
plt.legend(prop={'size':18})
plt.ylim([-1.0,1.0])

# <headingcell level=2>

# State Vector

# <codecell>

fig = plt.figure(figsize=(16,2*numstates))

# X/Y
plt.subplot(611)
plt.step(range(len(measurements[0])),x0-mx[0], label='$x$')
plt.step(range(len(measurements[0])),x1-my[0], label='$y$')

plt.title('Extended Kalman Filter State Estimates (State Vector $x$)')
plt.legend(loc='best',prop={'size':22})
plt.ylabel('Position [m]')

# Course
plt.subplot(612)
plt.step(range(len(measurements[0])),np.multiply(x2,180.0/np.pi), label='$\psi$')
plt.step(range(len(measurements[0])),(course+180.0)%(360.0)-180.0, label='$\psi$ (GPS)', alpha=0.5)
plt.ylabel('Course $^\circ$')
plt.yticks(np.arange(-180, 181, 45))
plt.ylim([-200,200])
plt.legend(loc='best',prop={'size':16})

# Velocity
plt.subplot(613)
plt.step(range(len(measurements[0])),np.multiply(x3,3.6), label='$v$')
plt.step(range(len(measurements[0])),speed, label='$v$ (GPS)', alpha=0.5)
plt.ylabel('Velocity $km/h$')
#plt.ylim([0, 30])
plt.legend(loc='best',prop={'size':16})

# Yawrate
plt.subplot(614)
plt.step(range(len(measurements[0])),np.multiply(x4,180.0/np.pi), label='$\dot \psi$')
plt.step(range(len(measurements[0])),yawrate, label='$\dot \psi$ (IMU)', alpha=0.5)
plt.ylabel('Yaw Rate $^\circ/s$')
#plt.ylim([-0.6, 0.6])
plt.legend(loc='best',prop={'size':16})

# Pitch
plt.subplot(615)
plt.step(range(len(measurements[0])),np.multiply(x5,180.0/np.pi), label='$\phi$')
#plt.step(range(len(measurements[0])),pitchacc*180.0/np.pi, label='$\phi$ (Acc)', alpha=0.5)
plt.step(range(len(measurements[0])),pitchrate, label='$\dot \phi$ (IMU)', alpha=0.5)
plt.ylabel('Pitch$^\circ$')
plt.ylim([-20.0, 20.0])
plt.legend(loc='best',prop={'size':16})
plt.twinx()
plt.step(range(len(measurements[0])),altitude, label='$H$ (GPS)', alpha=0.2, color='k')
plt.ylabel('Altitude $m$')
plt.ylim([100.0, 140.0])


# Roll
plt.subplot(616)
plt.step(range(len(measurements[0])),np.multiply(x7,180.0/np.pi), label='$\Theta$')
#plt.step(range(len(measurements[0])),rollacc*180.0/np.pi, label='$\Theta$ (Acc)', alpha=0.5)
plt.step(range(len(measurements[0])),rollrate, label='$\dot \Theta$ (IMU)', alpha=0.5)
plt.ylabel('Roll$^\circ$')
plt.ylim([-20.0, 20.0])
plt.legend(loc='best',prop={'size':16})
plt.xlabel('Filter Step')

plt.savefig('Extended-Kalman-Filter-CTRV-Attitude-State-Estimates.png', dpi=72, transparent=True, bbox_inches='tight')

# <codecell>


# <headingcell level=2>

# Position x/y

# <codecell>

%pylab --no-import-all

# <codecell>

fig = plt.figure(figsize=(16,9))

# EKF State
qscale= 0.5*np.divide(x3,np.max(x3))+0.1
plt.quiver(x0,x1,np.cos(x2), np.sin(x2), color='#94C600', units='xy', width=0.01, scale=qscale)
plt.scatter(x0,x1, c=dstate, s=30, label='EKF Position')

# Measurements
plt.scatter(mx[::5],my[::5], s=50, label='GPS Measurements', c=sp[::5], cmap='autumn_r',norm=matplotlib.colors.LogNorm())
cbar=plt.colorbar()
cbar.ax.set_ylabel(u'$\sigma^2_r$ (Elements of Measurement Noise Covariance Matrix $R$)', rotation=270)
cbar.ax.set_xlabel(u'm')

# Start/Goal
plt.scatter(x0[0],x1[0], s=60, label='Start', c='g')
plt.scatter(x0[-1],x1[-1], s=60, label='Goal', c='r')

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Position')
#plt.legend(loc='best')
plt.axis('equal')
#plt.tight_layout()

plt.show()
#plt.savefig('Extended-Kalman-Filter-CTRV-Position.png', dpi=72, transparent=True, bbox_inches='tight')

# <codecell>

print('Done.')

# <headingcell level=1>

# Conclusion

# <markdowncell>

# As you can see, complicated analytic calculation of the Jacobian Matrices, but it works pretty well.
# 
# Let's take a look at the trajectory on Google Earth:

# <headingcell level=2>

# Write Google Earth KML

# <headingcell level=3>

# Convert back from Meters to Lat/Lon (WGS84)

# <codecell>

latekf = latitude[0] + np.divide(x1,arc)
lonekf = longitude[0]+ np.divide(x0,np.multiply(arc,np.cos(latitude*np.pi/180.0)))

# <headingcell level=3>

# Create Data for KML Path

# <markdowncell>

# Coordinates and timestamps to be used to locate the car model in time and space
# The value can be expressed as yyyy-mm-ddThh:mm:sszzzzzz, where T is the separator between the date and the time, and the time zone is either Z (for UTC) or zzzzzz, which represents ±hh:mm in relation to UTC.

# <codecell>

import datetime
car={}
car['when']=[]
car['coord']=[]
car['gps']=[]
for i in range(len(millis)):
    d=datetime.datetime.fromtimestamp(millis[i]/1000.0)
    car["when"].append(d.strftime("%Y-%m-%dT%H:%M:%SZ"))
    car["coord"].append((lonekf[i], latekf[i], 0))
    car["gps"].append((longitude[i], latitude[i], 0))

# <codecell>

from simplekml import Kml, Model, AltitudeMode, Orientation, Scale

# <codecell>

# The model path and scale variables
car_dae = r'http://simplekml.googlecode.com/hg/samples/resources/car-model.dae'
car_scale = 1.0

# Create the KML document
kml = Kml(name=d.strftime("%Y-%m-%d %H:%M"), open=1)

# Create the model
model_car = Model(altitudemode=AltitudeMode.clamptoground,
                            orientation=Orientation(heading=75.0),
                            scale=Scale(x=car_scale, y=car_scale, z=car_scale))

# Create the track
trk = kml.newgxtrack(name="EKF", altitudemode=AltitudeMode.clamptoground,
                     description="State Estimation from Extended Kalman Filter with CTRV Model")
gps = kml.newgxtrack(name="GPS", altitudemode=AltitudeMode.clamptoground,
                     description="Original GPS Measurements")

# Attach the model to the track
trk.model = model_car
gps.model = model_car

trk.model.link.href = car_dae
gps.model.link.href = car_dae

# Add all the information to the track
trk.newwhen(car["when"])
trk.newgxcoord(car["coord"])

gps.newwhen(car["when"][::5])
gps.newgxcoord((car["gps"][::5]))

# Style of the Track
trk.iconstyle.icon.href = ""
trk.labelstyle.scale = 1
trk.linestyle.width = 10
trk.linestyle.color = '7f00ff00' # aabbggrr

gps.iconstyle.icon.href = ""
gps.labelstyle.scale = 0
gps.linestyle.width = 4
gps.linestyle.color = '7fff0000'


# Saving
#kml.save("Extended-Kalman-Filter-CTRV.kml")
kml.savekmz("Extended-Kalman-Filter-CTRV-Adaptive.kmz")

# <codecell>

print('Exported KMZ File for Google Earth')

# <headingcell level=2>

# Screenshot

# <markdowncell>

# ![Google Earth](https://raw.github.com/balzer82/Kalman/master/Extended-Kalman-Filter-CTRV-Adaptive-Kurve.jpg)

# <markdowncell>

# Works just fine!

