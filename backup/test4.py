from your_package import embedding
import numpy as np



points = np.random.rand(5, 10) / 10
hyperbolic_space_poincare = embedding.HyperbolicSpace(model='poincare', curvature=-1, points=points)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.model)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.points)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.dimension)


hyperbolic_space_poincare.switch_model('loid')
# print(hyperbolic_space_poincare.centroid())

# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.model)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.points)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.dimension)

hyperbolic_space_poincare.switch_model('poincare')
print(hyperbolic_space_poincare.points)
print(hyperbolic_space_poincare.centroid())
hyperbolic_space_poincare.center()
print(hyperbolic_space_poincare.points)
print(hyperbolic_space_poincare.centroid())


# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.model)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.points)
# print("Switched model to poincare in hyperbolic space:", hyperbolic_space_poincare.dimension)