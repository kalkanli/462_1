import numpy as np

def sample_sphere(r=1, npoints=1000):
    dr = np.random.rand(npoints,1)
    theta = 2*np.pi*np.random.rand(npoints,1)
    x = dr * np.cos(theta)
    y = dr * np.sin(theta)
    return np.hstack((x,y))

def create_dataset(npoints=1000):
    points = sample_sphere(npoints=npoints) 
    points = np.hstack((np.ones(npoints).reshape(npoints,1), points))
    labels = np.sign(points[:, 2])
    return points, labels