from your_package import tree_processor
import numpy as np

# Example 1: Initializing MultiTree with a file path
file_path = 'your_package/tree/filtered_filterln3-fna2aa.tre'
multi_tree = tree_processor.MultiTree(file_path, enable_logging=True)



# Example 2: Initializing MultiTree with a name and a set of Tree objects

tree1 = tree_processor.Tree("your_package/tree/filtered_filterln3-fna2aa_trees/tree_1.tre")
tree2 = tree_processor.Tree("your_package/tree/filtered_filterln3-fna2aa_trees/tree_2.tre")

trees = {tree1, tree2}
multi_tree = tree_processor.MultiTree("ExampleTreeSet", trees, enable_logging=True)

# Example 3: Making a copy of a MultiTree object
copied_multi_tree = multi_tree.copy()

# Example 4: Loading trees from a file
try:
    multi_tree = tree_processor.MultiTree(file_path, enable_logging=True)
except FileNotFoundError as e:
    print(f"Error: {e}")

# # Example 5: Saving trees to a file
# multi_tree.save_trees('your_package/tree/saved_trees_file.newick')

# # Example 6: Computing a distance matrix
# distance_matrix, confidence_matrix = multi_tree.distance_matrix(func=np.nanmean, confidence=True)
# print("Distance Matrix:\n", distance_matrix)
# print("Confidence Matrix:\n", confidence_matrix)

# Example 7: Adding metadata
multi_tree.add_metadata("author", "John Doe")
multi_tree.add_metadata("description", "A collection of example trees")
print(multi_tree)

# Example 8: Searching for trees by name
result_trees = multi_tree.search_trees("tree_1")
print("Search Results:", result_trees)

# Example 9: Iterating over trees
for tree in multi_tree:
    print("Tree:", tree)

# Example 10: Checking the number of trees
print("Number of trees:", len(multi_tree))

# Example 11: Checking if a tree is in the collection
is_in_collection = tree_processor.Tree("tree_1", tree1.contents) in multi_tree
print("Is tree1 in collection?", is_in_collection)

# Example 12: Using __repr__ for debugging
print(repr(multi_tree))
