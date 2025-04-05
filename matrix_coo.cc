#include "matrix_coo.hh"
extern "C" {
#include "mmio.h"
}
#include <mpi.h>
#include <unordered_map>
#include "utils.cpp"

void MatrixCOO::read(const std::string & fn) {
  int nz;

  int ret_code;
  MM_typecode matcode;
  FILE * f;

  if ((f = fopen(fn.c_str(), "r")) == NULL) {
    printf("Could not open matrix");
    exit(1);
  }

  if (mm_read_banner(f, &matcode) != 0) {
    printf("Could not process Matrix Market banner.\n");
    exit(1);
  }

  // Matrix is sparse
  if (not(mm_is_matrix(matcode) and mm_is_coordinate(matcode))) {
    printf("Sorry, this application does not support ");
    printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
    exit(1);
  }

  if ((ret_code = mm_read_mtx_crd_size(f, &m_m, &m_n, &nz)) != 0) {
    exit(1);
  }

  /* reserve memory for matrices */
  irn.resize(nz);
  jcn.resize(nz);
  a.resize(nz);

  /*  NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
  /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
  /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
  m_is_sym = mm_is_symmetric(matcode);
  for (int i = 0; i < nz; i++) {
    int I, J;
    double a_;

    fscanf(f, "%d %d %lg\n", &I, &J, &a_);
    I--; /* adjust from 1-based to 0-based */
    J--;

    irn[i] = I;
    jcn[i] = J;
    a[i] = a_;
  }

  if (f != stdin) {
    fclose(f);
  }
}

MatrixCOO::MatrixCOO(const std::vector<double>& values, const std::vector<int>& row_indices, const std::vector<int>& col_indices)
{
  // Copy the distributed data to the base class vectors
  a = values;
  irn = row_indices;
  jcn = col_indices;
  
  // We need to determine matrix dimensions from the indices
  if (!irn.empty()) {
    m_m = *std::max_element(irn.begin(), irn.end()) + 1;
    m_n = *std::max_element(jcn.begin(), jcn.end()) + 1;
  } else {
    m_m = 0;
    m_n = 0;
  }
  
  // Symmetry is false because we use this constructor for the distributed matrix
  m_is_sym = false;
}

void MatrixCOO::get_displs_sdcts(const std::vector<int>& irn, const int size, std::vector<int>& displs, std::vector<int>& sendcounts){

  //get number of elements per row, for example A=[0,0, 1,1,1, 2,2, 3,3, 4,4] will yield 
  //[[0,2],
  // [1,3],
  // [2,2],
  // [3,2],
  // [4,2]]
  std::unordered_map <std::size_t, int> row_counts;
  for (std::size_t idx : irn) {
      row_counts[idx]++;
  }

  const int n = row_counts.size();
  std::vector<int> row_starts(n + 1, 0);
  int cumulative_sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
      cumulative_sum += row_counts[i];
      row_starts[i + 1] = cumulative_sum;
  }
  //gives back at which index each sequence of elts starts, i.e for A we'll get [0,2,5,7,9,11]
  
  const int chunk_size = n / size;
  const int remainder = n % size;
  int start, end;

  for(int rank = 0; rank < size; ++rank){
  
    if (rank < remainder) {
        start = rank * (chunk_size + 1);
        end = start + chunk_size + 1;
    } else {
        start = remainder * (chunk_size + 1) + (rank - remainder) * chunk_size;
        end = start + chunk_size;
    }
    //std::cout << "Rank " << rank << " gets rows from " << row_starts[start] << " to " << row_starts[end] << std::endl;

    displs[rank] = row_starts[start];
    sendcounts[rank] = row_starts[end] - row_starts[start];
  }
}


void MatrixCOO::distribute_mat(std::vector<double>& distrib_a, std::vector<int>& distrib_irn, 
  std::vector<int>& distrib_jrn, const int size, const int rank) {
  
  // If matrix is symmetric, add the missing elements
  if (m_is_sym) {
    expand_indices(irn, jcn, a);
  }

  //Declare variables for Scatterv
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  //reorder vectors
  const std::vector<std::size_t> indices = argsort(irn);
  const std::vector<int> ordered_irn = reorderVector(irn, indices);
  const std::vector<int> ordered_jcn = reorderVector(jcn, indices);
  const std::vector<double> ordered_a = reorderVector(a, indices);

  
  // Calculate distribution sizes  
  if (rank == 0){
    get_displs_sdcts(ordered_irn, size, displs, sendcounts);}

  //Make sure the receive buffers have the correct size
  MPI_Bcast(sendcounts.data(), sendcounts.size(), MPI_INT, 0, MPI_COMM_WORLD);
  const int count_recv = sendcounts[rank]
  distrib_a.resize(count_recv);
  distrib_irn.resize(count_recv);
  distrib_jrn.resize(count_recv);

  //Scatter the data
  MPI_Scatterv(ordered_a.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
  distrib_a.data(), count_recv, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(ordered_irn.data(), sendcounts.data(), displs.data(), MPI_INT,
  distrib_irn.data(), count_recv, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(ordered_jcn.data(), sendcounts.data(), displs.data(), MPI_INT,
  distrib_jrn.data(), count_recv, MPI_INT, 0, MPI_COMM_WORLD);
}
