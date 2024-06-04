import numpy as np
from scipy.stats import linregress

def pointRegression(points: np.ndarray[tuple[int, int]]):
    nlist = points.tolist()
    xs = list(map(lambda x: x[0], nlist))
    ys = list(map(lambda x: x[1], nlist))
    
    result = linregress(xs, ys)
    
    return result.slope