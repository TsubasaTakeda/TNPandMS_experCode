from scipy import io

def write_sparse_mat(file_path, sparse_mat):

    io.mmwrite(file_path, sparse_mat)


def read_sparse_mat(file_path):

    sparse_mat = io.mmread(file_path)

    return sparse_mat

