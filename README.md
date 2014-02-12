Multidimensional Kalman-Filter
==============================
Some Python Implementations of the Kalman Filter
------------------------------

![Kalman Filter Step](http://bilgin.esme.org/portals/0/images/kalman/iteration_steps.gif)
`Image from bilgin.esme.org`

## Kalman Filter with Constant Velocity Model

[View Notebook](http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Kalman-Filter-CV.ipynb?create=1)

## Kalman Filter with Constant Acceleration Model

###[View iPython Notebook](http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Kalman-Filter-CA.ipynb?create=1)

Constant Acceleration Model for Ego Motion in Plane

![State Vector](http://latex.codecogs.com/gif.latex?x%3D%20%5Cbegin%7Bbmatrix%7D%20x%20%5C%5C%20y%20%5C%5C%20%5Cdot%20x%20%5C%5C%20%5Cdot%20y%20%5C%5C%20%5Cddot%20x%20%5C%5C%20%5Cddot%20y%20%5Cend%7Bbmatrix%7D)

### Covariance Matrix

![Covariance Matrix](https://raw.github.com/balzer82/Kalman/master/Kalman-Filter-CA-CovarianceMatrix.png)

### State Estimate

![State Vector x](https://raw.github.com/balzer82/Kalman/master/Kalman-Filter-CA-StateEstimated.png)

### Position

![Position](https://raw.github.com/balzer82/Kalman/master/Kalman-Filter-CA-Position.png)

## Adaptive Kalman Filter with Constant Velocity Model

[View Notebook](http://nbviewer.ipython.org/github/balzer82/Kalman/blob/master/Adaptive-Kalman-Filter-CV.ipynb?create=1)
