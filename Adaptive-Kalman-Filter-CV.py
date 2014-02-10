# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# <headingcell level=1>

# Adaptive Kalman Filter Implementation in Python

# <markdowncell>

# ![Image](http://www.cbcity.de/wp-content/uploads/2013/06/Fahrzeug_GPS_Tunnel-520x181.jpg)
# 
# Situation covered: You drive with your car in a tunnel and the GPS signal is lost. Now the car has to determine, where it is in the tunnel. The only information it has, is the velocity in driving direction. The x and y component of the velocity ($\dot x$ and $\dot y$) can be calculated from the absolute velocity (revolutions of the wheels) and the heading of the vehicle (yaw rate sensor).

# <headingcell level=2>

# State Vector

# <markdowncell>

# Constant Velocity Model for Ego Motion
# 
# $$x= \left[ \matrix{ x \\ y \\ \dot x \\ \dot y} \right]$$
# 

# <markdowncell>

# Formal Definition:
# 
# $$x_{k+1} = F \cdot x_{k}$$
# 
# $$x_{k+1} = \begin{bmatrix}1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \cdot \begin{bmatrix} x \\ y \\ \dot x \\ \dot y \end{bmatrix}_{k}$$
# 
# $$y = H \cdot x$$
# 
# $$y = \begin{bmatrix}0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1\end{bmatrix} \cdot x$$

# <headingcell level=4>

# Initial State

# <codecell>

x = np.matrix([[0.0, 0.0, 0.0, 0.0]]).T
print(x, x.shape)
plt.scatter(x[0],x[1], s=100)
plt.title('Initial Location')

# <headingcell level=4>

# Initial Uncertainty

# <codecell>

P = np.matrix([[0.1, 0.0, 0.0, 0.0],
              [0.0, 0.1, 0.0, 0.0],
              [0.0, 0.0, 100.0, 0.0],
              [0.0, 0.0, 0.0, 100.0]])
print(P, P.shape)


# Plot between -10 and 10 with .001 steps.
xpdf = np.arange(-10, 10, 0.001)

plt.subplot(221)
plt.plot(xpdf, norm.pdf(xpdf,0,P[0,0]))
plt.title('$x$')

plt.subplot(222)
plt.plot(xpdf, norm.pdf(xpdf,0,P[1,1]))
plt.title('$y$')

plt.subplot(223)
plt.plot(xpdf, norm.pdf(xpdf,0,P[2,2]))
plt.title('$\dot x$')

plt.subplot(224)
plt.plot(xpdf, norm.pdf(xpdf,0,P[3,3]))
plt.title('$\dot y$')
plt.tight_layout()

# <headingcell level=4>

# Dynamic Matrix

# <markdowncell>

# It is calculated from the dynamics of the Egomotion.
# 
# $$x_{k+1} = x_{k} + \dot x_{k} \cdot \Delta t$$
# $$y_{k+1} = y_{k} + \dot y_{k} \cdot \Delta t$$
# $$\dot x_{k+1} = \dot x_{k}$$
# $$\dot y_{k+1} = \dot y_{k}$$

# <codecell>

dt = 0.5 # Time Step between Filter Steps

F = np.matrix([[1.0, 0.0, dt, 0.0],
              [0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(F, F.shape)

# <headingcell level=4>

# Measurement Matrix

# <codecell>

H = np.matrix([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]])
print(H, H.shape)

# <headingcell level=4>

# Measurement Noise Covariance

# <markdowncell>

# R will be updated in each filter step in the adaptive Kalman Filter

# <codecell>

ra = 200.0

R = np.matrix([[ra, 0.0],
              [0.0, ra]])
print(R, R.shape)

plt.subplot(121)
plt.plot(xpdf, norm.pdf(xpdf,0,R[0,0]))
plt.title('$\dot x$')

plt.subplot(122)
plt.plot(xpdf, norm.pdf(xpdf,0,R[1,1]))
plt.title('$\dot y$')
plt.tight_layout()

# <headingcell level=4>

# Process Noise Covariance

# <codecell>

G = np.matrix([[dt],
               [dt],
               [0.0],
               [0.0]])
Q = G*G.T*5.0


plt.subplot(221)
plt.plot(xpdf, norm.pdf(xpdf,0,Q[0,0]))
plt.title('$x$')

plt.subplot(222)
plt.plot(xpdf, norm.pdf(xpdf,0,Q[1,1]))
plt.title('$y$')

plt.subplot(223)
plt.plot(xpdf, norm.pdf(xpdf,0,Q[2,2]))
plt.title('$\dot x$')

plt.subplot(224)
plt.plot(xpdf, norm.pdf(xpdf,0,Q[3,3]))
plt.title('$\dot y$')
plt.tight_layout()

# <headingcell level=4>

# Identity Matrix

# <codecell>

I = np.eye(4)
print(I, I.shape)

# <headingcell level=2>

# Measurement

# <codecell>

m = 200 # Measurements
vx= 10 # in X
vy= 10 # in Y

mx = np.array(vx+np.random.randn(m))
my = np.array(vy+np.random.randn(m))

# some different error somewhere in the measurements
my[(m/2):(3*m/4)]= np.array(vy+20.0*np.random.randn(m/4))

measurements = np.vstack((mx,my))

#print(measurements, measurements.shape)

# <codecell>

fig = plt.figure(figsize=(16,5))

plt.step(range(m),mx, label='$\dot x$')
plt.step(range(m),my, label='$\dot y$')
plt.ylabel('Velocity')
plt.title('Measurements')
plt.legend(loc='best',prop={'size':18})

# <codecell>

# Preallocation for Plotting
xt = []
yt = []
dxt= []
dyt= []
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Kx = []
Ky = []
Kdx= []
Kdy= []

# <headingcell level=2>

# Kalman Filter

# <markdowncell>

# ![Kalman Filter](http://bilgin.esme.org/portals/0/images/kalman/iteration_steps.gif)

# <codecell>

for n in range(len(measurements[0])):
    
    # Adaptive Measurement Covariance from last i Measurements
    i = 10
    j = 10.0 # Factor
    if n>i:
        R = np.matrix([[j*np.std(measurements[0,(n-i):n]), 0.0],
                      [0.0, j*np.std(measurements[1,(n-i):n])]])
    

    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.pinv(S)

    
    # Update the estimate via z
    Z = measurements[:,n].reshape(2,1)
    y = Z - (H*x)                            # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*H))*P

    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = F*x
    
    # Project the error covariance ahead
    P = F*P*F.T + Q
    
    
    # Save states for Plotting
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))    

# <codecell>


# <headingcell level=2>

# Plots

# <headingcell level=3>

# Unsicherheiten

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.plot(range(len(measurements[0])),Px, label='$x$')
plt.plot(range(len(measurements[0])),Py, label='$y$')
plt.plot(range(len(measurements[0])),Pdx, label='$\dot x$')
plt.plot(range(len(measurements[0])),Pdy, label='$\dot y$')

plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Uncertainty (Elements from Matrix $P$)')
plt.legend(loc='best',prop={'size':22})

# <headingcell level=3>

# Kalman Gains

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.plot(range(len(measurements[0])),Kx, label='Kalman Gain for $x$')
plt.plot(range(len(measurements[0])),Ky, label='Kalman Gain for $y$')
plt.plot(range(len(measurements[0])),Kdx, label='Kalman Gain for $\dot x$')
plt.plot(range(len(measurements[0])),Kdy, label='Kalman Gain for $\dot y$')

plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Kalman Gain (the lower, the more the measurement fullfill the prediction)')
plt.legend(loc='best',prop={'size':22})

# <headingcell level=3>

# Covariance Matrix

# <codecell>

fig = plt.figure(figsize=(5, 5))
im = plt.imshow(P, interpolation="none")
plt.title('Covariance Matrix $P$')
ylocs, ylabels = yticks()
# set the locations of the yticks
yticks(arange(5))
# set the locations and labels of the yticks
yticks(arange(4),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

xlocs, xlabels = xticks()
# set the locations of the yticks
xticks(arange(5))
# set the locations and labels of the yticks
xticks(arange(4),('$x$', '$y$', '$\dot x$', '$\dot y$'), fontsize=22)

plt.xlim([-0.5,3.5])
plt.ylim([3.5, -0.5])

from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", "5%", pad="3%")
plt.colorbar(im, cax=cax)


plt.tight_layout()

# <headingcell level=3>

# Measurements

# <codecell>

fig = plt.figure(figsize=(16,9))
plt.step(range(len(measurements[0])),dxt, label='$\dot x$')
plt.step(range(len(measurements[0])),dyt, label='$\dot y$')

plt.xlabel('Filter Step')
plt.ylabel('')
plt.title('Estimate (Elements from State Vector $x$)')
plt.legend(loc='best',prop={'size':22})
plt.ylabel('Geschwindigkeit')

# <codecell>

fig = plt.figure(figsize=(16,16))
plt.scatter(xt,yt, s=20, label='State', c='k')
plt.scatter(xt[0],yt[0], s=100, label='Start', c='g')
plt.scatter(xt[-1],yt[-1], s=100, label='Goal', c='r')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Position')
plt.legend(loc='best')
axis('equal')

# <headingcell level=1>

# Conclusion

# <markdowncell>

# As you can see, between Filter Step 150 and 200 you have massive noise in the measurement, but the adaptive filter is raising the measurement covariance, so that the Kalman Gain is tryin to use the dynamic instead of the measurement

