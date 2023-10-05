import argparse
import scipy.io as sio
import scipy.sparse as sps
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description='Convert a directed Matrix Market file to an undirected graph, remove the weight column, and add 1 to every row in the CSV.')
    parser.add_argument('input_file', help='Path to the input directed Matrix Market file')
    parser.add_argument('output_file_mm', help='Path to save the output undirected Matrix Market file without the weight column')

    args = parser.parse_args()

    input_file_path = args.input_file
    output_file_mm_path = args.output_file_mm

    # Load the directed matrix from the input file
    directed_matrix = sio.mmread(input_file_path).tocoo()

    # Create the undirected matrix by adding the original matrix to its transpose
    undirected_matrix = directed_matrix + directed_matrix.transpose()

    # Convert the undirected matrix to CSR format
    undirected_matrix_csr = undirected_matrix.tocsr()

    # Remove the lower triangular part by slicing
    undirected_matrix_csr = undirected_matrix_csr[:, :undirected_matrix_csr.shape[0]]

    # Save the undirected matrix as a Matrix Market file
    sio.mmwrite(output_file_mm_path, undirected_matrix_csr, field="pattern")


if __name__ == "__main__":
    main()
