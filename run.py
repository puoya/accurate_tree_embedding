import os
import sys
import argparse
import logging
from datetime import datetime
import spaceform_pca_lib as sfpca


def setup_logging(dataset_name):
    # Define log directory and create it if it doesn't exist
    log_directory = f"datasets/{dataset_name}/log_files"
    os.makedirs(log_directory, exist_ok=True)

    # Get the current time for log filename
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_directory, f"exp_{current_time}.log")

    # Set up logging
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s:%(levelname)s:%(message)s')

def main(args):
    dataset_name = args.dataset
    method = args.model
    dimension = args.dimension

    # Check if dataset directory exists
    if not os.path.exists(f"datasets/{dataset_name}"):
        sys.exit(f"Dataset directory 'datasets/{dataset_name}' does not exist. Exiting...")

    # Set up logging
    setup_logging(dataset_name)

    # Perform tasks based on the specified method
    if method in ['sfpca','horopca', 'bsa', 'pga']:
        # sfpca.extract_trees(dataset_name)
        # sfpca.process_all_trees(dataset_name)
        sfpca.hyperbolic_mds(dataset_name,dimension)
        sfpca.precise_hyperbolic_mds(dataset_name,dimension)

    else:
        logging.error(f"Method '{method}' is not supported.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with different models on datasets.')
    parser.add_argument('--dataset', type=str, default='unfiltered_sorted', help='Name of the dataset (default: unfiltered_sorted)')
    parser.add_argument('--model', type=str, choices=['sfpca', 'horopca', 'bsa', 'pga'], 
                        default='sfpca', help='Model to run (default: sfpca)')
    parser.add_argument('--dimension', type=int, default=10, help='Dimension parameter (default: 10)')
    
    args = parser.parse_args()
    main(args)
