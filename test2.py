import numpy as np
from scipy.stats import linregress

def pointRegression(points: np.ndarray[tuple[int, int]], yoffset: int):
    nlist = points.tolist()
    xs = list(map(lambda x: x[0], nlist))
    ys = list(map(lambda x: yoffset - x[1], nlist))
    
    result = linregress(xs, ys)
    
    return result.slope