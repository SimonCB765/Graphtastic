import scipy.spatial

def euclidean(x, y):
    return scipy.spatial.distance.euclidean(x, y)

metrics = {
           'Euclidean' : euclidean
          }