from your_package import tree_processor
import numpy as np


# T1 = tree_processor.Tree()


# T1 = tree_processor.Tree("your_package/tree/Best.FAA.tre")
MT = tree_processor.MultiTree("your_package/tree/Best.FAA.tre")
T1 = MT.trees[0]


# T1.name = 'large_tree'
# print(T1)
# print(T1.name)
# print(T1.contents)

# T2 = tree_processor.Tree("your_package/tree/filtered_filterln3-fna2aa_trees/tree_2.tre")
# print(T2)
# print(T2.name)
# print(T2.contents)

# timestamp = '20240731_081322'
# output_dir = f'your_package/images_tmp/{timestamp}'

TP = tree_processor.TreeProcessor(T1)
D = TP.hyperbolic_embedding(dimension = 50, mode = 'precise',initial_lr = 10**(-5))

# TP.create_movie(output_dir,fps = 33)

