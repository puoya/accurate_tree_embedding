from tree_processor import TreeProcessor

# Create an instance of TreeProcessor
processor = TreeProcessor()

# Set the source directory and output directory
# processor.source_directory("my_tree_files")
# processor.output_directory("my_extracted_trees")

# Extract trees from the source directory to the output directory
processor.extract_trees()

# Example output for the user to understand usage
print("Tree extraction complete.")
