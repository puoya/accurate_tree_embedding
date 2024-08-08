from your_package import embedding
from your_package import procrustes
import numpy as np
import torch
import logging
from your_package import __version__
logging.basicConfig(filename='log/package_log.log', level=logging.INFO)
logging.info(f'Running version {__version__}')





initial_points = np.array([
    [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
    [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
])
# initial_points = np.random.randn(2, 30)
initial_points  =initial_points/20
space1 = embedding.PoincareEmbedding(points = initial_points)
b = torch.tensor(
    [1,0.1])
b = b / 10
# space1.translate(b)


initial_points = np.array([
    [1, -2, 1, 0, .1, 1, -4, 8.1, 9.1, 1.3],
    [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
])
initial_points = initial_points/20
# initial_points  = initial_points + np.random.randn(2, 30)/ 30 
# space2 = space1.copy()
space2 = embedding.PoincareEmbedding(points = initial_points)



R = np.random.randn(2,2)
R = np.matmul(R.T,R)
R,_,_ = np.linalg.svd(R,full_matrices=True)
space2.rotate(R)
b = torch.tensor(
    [-1,1.1])
b = b / 30
# space2.translate(b)
# space2 = space2.switch_model()
print('space2', space2.points)




hp = procrustes.HyperbolicProcrustes(space1, space2, mode = 'accurate', enable_logging = True)
space3 = hp.map(space1.switch_model())
print('space3', space3.points)


space3 = hp.map(space1)
print('space4', space3.points)