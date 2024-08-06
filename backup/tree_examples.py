# Import necessary libraries
import numpy as np
from your_package import tree_processor

# Example 1: Initializing a Tree from a File
# Initialize a Tree from a Newick file
tree = tree_processor.Tree("your_package/tree/filtered_filterln3-fna2aa_trees/tree_1.tre")
print(tree)

# Example 2: Initializing a Tree with Contents
# Initialize a Tree with a name and a treeswift tree object
tree = tree_processor.Tree("Example Tree", tree.contents)
print(tree)

# Example 3: Enabling Logging
# Initialize a Tree from a file with logging enabled
tree_with_logging = tree_processor.Tree("your_package/tree/filtered_filterln3-fna2aa_trees/tree_1.tre", enable_logging=True)
print(tree_with_logging)

# Example 4: Using `from_contents` Class Method
# Create a Tree from contents using the class method
tree_from_contents = tree_processor.Tree("Example Tree", tree.contents, enable_logging=True)
print(tree_from_contents)

# Example 5: Copying a Tree
# Copy an existing Tree
tree_copy = tree.copy()
print(tree_copy)

tree_copy = tree_with_logging.copy()
print(tree_copy)


tree_copy = tree_from_contents.copy()
print(tree_copy)

# Example 6: Saving a Tree
# Save the Tree to a file in Newick format
tree.save_tree("your_package/tree/filtered_filterln3-fna2aa_trees/saved_tree.newick")

# Example 6: Saving a Tree
# Save the Tree to a file in Newick format
tree_copy.save_tree("your_package/tree/filtered_filterln3-fna2aa_trees/saved_tree_copy.newick")

# Example 7: Getting Terminal Names
# Get terminal (leaf) names in the Tree
terminal_names = tree_copy.terminal_names()
print(terminal_names)

# Example 8: Computing the Distance Matrix
# Compute the distance matrix for the Tree
distance_matrix = tree_copy.distance_matrix()
print(distance_matrix)

# Example 9: Computing the Diameter
# Compute the diameter of the Tree
diameter = tree_copy.diameter()
print(f"Diameter: {diameter}")

# Example 10: Normalizing the Tree
# Normalize the Tree so that the diameter is 1
tree_copy.normalize()
