from your_package import embedding
from your_package import procrustes
import numpy as np
import torch
import logging
from your_package import __version__
logging.basicConfig(filename='log/package_log.log', level=logging.INFO)
logging.info(f'Running version {__version__}')

###############################################
# test initialization
###############################################
###############################################
# space = embedding.Embedding()
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
#############################
# test complete (general)
#############################
# points = np.array([[1,2],[2,3]])
# space = embedding.Embedding(points = points)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
# points = torch.tensor([[1,2],[2,3]])
# space = embedding.Embedding(points = points)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
#############################
# test complete (np and torch points)
#############################
# points = np.array([[1,2],[2,3]])
# space = embedding.Embedding(points = points, geometry = 'euclidean',enable_logging = True)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# space3 = space2.copy()
# print(space2)
# print(space3)
#############################
# test complete (geometry, logger, copy)
#############################
# points = torch.tensor([[1,2],[2,3]])
# print(points)
# points = points.int()
# print(points)
# space = embedding.Embedding(points = points, geometry = 'euclidean',enable_logging = True)
# print('space',space)
# print('points',space.points)
# pt = space.points
# print(pt.dtype)
# print('log',space.logger)
# print('geom',space.geometry)
#############################
# test complete (torch dtype)
#############################






###############################################
# test initialization
###############################################
###############################################
# space = embedding.HyperbolicEmbedding()
# print(space)
# print(space.curvature)
# print(space.geometry)
# print(space.points)
# print(space.logger)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
# print(space2.copy())
#############################
# test complete (general)
#############################
# points = np.array([[1,2],[2,3.0]])/ 20 
# print(points)
# space = embedding.HyperbolicEmbedding(points = points)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
# points = torch.tensor([[1,2],[2,3]])
# space = embedding.HyperbolicEmbedding(points = points)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
#############################
# test complete (np and torch points)
#############################
# points = np.array([[1,2],[2,3]])
# space = embedding.HyperbolicEmbedding(points = points, curvature = -10, enable_logging = True)
# print('space',space)
# print('points',space.points)
# print('log',space.logger)
# print('curvature',space.curvature)
# space.curvature = -6
# print('curvature',space.curvature)
# print('geom',space.geometry)
# space.save('tmp.pkl')
# space2 = space.load('tmp.pkl')
# print(space2)
#############################
# test complete (curvature and logger)
#############################
# points = torch.tensor([[1,2],[2,3]])
# print(points)
# points = points.int()
# print(points)
# space = embedding.HyperbolicEmbedding(points = points, enable_logging = True,model = 'loid')
# print('space',space)
# print('points',space.points)
# pt = space.points
# print(pt.dtype)
# print('log',space.logger)
# print('geom',space.geometry)
# print('model',space.model)
#############################
# test complete (torch dtype, model)
#############################







###############################################
# test initialization
###############################################
###############################################
# space = embedding.PoincareEmbedding()
# print(space)
# print(space.geometry)
# print(space.points)
# print(space.dimension)
# print(space.model)
# print(space.curvature)
# print(space.logger)
# space = embedding.PoincareEmbedding(enable_logging = True)
# print(space.logger)
#############################
# test complete (general, logger)
#############################
# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# print(space)
# print(space.geometry)
# print(space.points)
# print(space.n_points)
# print(space.dimension)
# print(space.model)
# print(space.curvature)
# print(space.norm2())
# points = np.array([
#     [ -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [ -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])/20
# space.curvature= -10
# space.points = points
# print(space.points)
# print(space.n_points)
# print(space.dimension)
# print(space.curvature)
# points = np.array([[1,2],[2,3]])
# space.points = points
# print(space.points)
#############################
# test complete (points, norm2, dimension, n_points,curvature)
#############################
# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points, enable_logging = True)
# # print('points',space.points)
# space_l = space.switch_model()
# print(space_l)
# print('points',space_l.points)
# print('dimension',space_l.dimension)
# print('n_points',space_l.n_points)
# print('curvature',space_l.curvature)
# space_p = space_l.switch_model()
# print(space_p)
# print('points',space_p.points)
# print('dimension',space_p.dimension)
# print('n_points',space_p.n_points)
# print('curvature',space_p.curvature)
# space_p.curvature = -1.3
# print(space_p.curvature)
#############################
# test complete (switch model)
#############################
# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# distance_matrix = space.distance_matrix()
# print(distance_matrix)
# space.curvature = -9
# distance_matrix2 = space.distance_matrix()
# print(space.curvature)
# print(distance_matrix2/distance_matrix)
#############################
# test complete (distance matrix)
#############################
# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# print(space.points)
# distance_matrix = space.distance_matrix()
# # print(distance_matrix)

# R = np.random.randn(2,2)
# R = np.matmul(R.T,R)
# R,_,_ = np.linalg.svd(R,full_matrices=True)
# space.rotate(R)
# print(space.points)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix2-distance_matrix)
#############################
# test complete (rotation)
#############################
# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# pts = space.points
# print(pts)
# distance_matrix = space.distance_matrix()
# # print(distance_matrix)

# b = torch.tensor(
#     [1,0.1])
# b = b / 10
# space.translate(b)
# print(space.points)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix2-distance_matrix)
# space.translate(-b)
# print(space.points-pts)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix-distance_matrix2)

# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# c = space.centroid()
# print(c)
# c = space.centroid(mode = 'Frechet')
# print(c)

# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# # mode = 'defualt'
# mode = 'Frechet'
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# print(space.points)
# print(space.centroid())
# space.center()
# print(space.points)
# print(space.centroid())

#############################
# test complete (translation, centroid, center)
#############################




# initial_points = np.array([
#     [1, -2, 1, 0, .1, 1, -4, 8, 9, 1.3],
#     [.1, -1.2, 2, 0.3, .4, -3.1, 2.4, 7.8, 4.9, -1]
# ])
# initial_points  =initial_points/20
# space = embedding.PoincareEmbedding(points = initial_points)
# space = space.switch_model()
# initial_points = space.points
# print(initial_points)
# print(space)
# print(space.geometry)
# print(space.points)
# print(space.dimension)
# print(space.model)
# print(space.curvature)
# print(space.logger)
# space = embedding.LoidEmbedding(enable_logging = True)
# print(space.logger)
# print(space.points)
# print(space.dimension)
# print(space.n_points)
# print(space.curvature)
###############################################
# test initialization
###############################################
###############################################

#############################
# test complete (general, logger)
#############################
# space = embedding.LoidEmbedding(points = initial_points)
# print(space)
# print(space.geometry)
# print(space.points)
# print(space.n_points)
# print(space.dimension)
# print(space.model)
# print(space.curvature)
# print(space.norm2())

# initial_points2 = initial_points[:,1:]
# space.curvature= -10
# space.points = initial_points2
# print(space.points)
# print(space.n_points)
# print(space.dimension)
# print(space.curvature)
# points = np.array([[1,2],[2,3]])
# space.points = points
# print(space)
# print(space.points)
#############################
# test complete (points, norm2, dimension, n_points,curvature)
#############################
# space = embedding.LoidEmbedding(points = initial_points, enable_logging = True)
# print('points',space.points)
# space_p = space.switch_model()
# print(space_p)
# print('points',space_p.points)
# print('dimension',space_p.dimension)
# print('n_points',space_p.n_points)
# print('curvature',space_p.curvature)
# space_l = space_p.switch_model()
# print(space_l)
# print('points',space_l.points)
# print('dimension',space_l.dimension)
# print('n_points',space_l.n_points)
# print('curvature',space_l.curvature)
# space_l.curvature = -1.3
# print(space_l.curvature)
#############################
# test complete (switch model)
#############################
# space = embedding.LoidEmbedding(points = initial_points)
# distance_matrix = space.distance_matrix()
# print(distance_matrix)
# space.curvature = -9
# distance_matrix2 = space.distance_matrix()
# print(space.curvature)
# print(distance_matrix2/distance_matrix)
#############################
# test complete (distance matrix)
#############################
# space = embedding.LoidEmbedding(points = initial_points)
# # print(space.points)
# distance_matrix = space.distance_matrix()
# # print(distance_matrix)
# # print(space.dimension)
# R = np.random.randn(2,2)
# R = np.matmul(R.T,R)
# R_,_,_ = np.linalg.svd(R,full_matrices=True)
# R = np.zeros((3,3))
# R[0,0] = 1
# R[1:,1:] = R_
# space.rotate(R)
# # print(space.points)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix2-distance_matrix)
#############################
# test complete (rotation)
#############################
# space = embedding.LoidEmbedding(points = initial_points)
# pts = space.points
# print(pts)
# distance_matrix = space.distance_matrix()
# # print(distance_matrix)

# b = torch.tensor(
#     [0,10,-18])
# b = b / 10
# b[0] = torch.sqrt(1+torch.sum(b**2))
# space.translate(b)
# # print(space.points)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix2-distance_matrix)
# b = torch.tensor(
#     [0,10,-18])
# b = -b / 10
# b[0] = torch.sqrt(1+torch.sum(b**2))
# space.translate(b)
# print(space.points-pts)
# distance_matrix2 = space.distance_matrix()
# print(distance_matrix-distance_matrix2)


# c = space.centroid()
# print(c)
# c = space.centroid(mode = 'Frechet')
# print(c)
# print(space.to_poincare(c))

# mode = 'default'
# # mode = 'Frechet'
# space = embedding.LoidEmbedding(points = initial_points)
# print(space.points)
# print(space.centroid(mode = mode))
# space.center(mode = mode)
# print(space.points)
# print(space.centroid(mode = mode))

#############################
# test complete (translation, centroid, center)
#############################
