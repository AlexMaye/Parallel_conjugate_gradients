#include "matrix_coo.hh"
extern "C" {
#include "mmio.h"
}
#include <mpi.h>

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

MatrixCOO::MatrixCOO(std::vector<double>& values, std::vector<int>& row_indices, std::vector<int>& col_indices)
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
  m_is_sym = false;  // Default to non-symmetric
}


void MatrixCOO::distribute_mat(std::vector<double>& distrib_a, std::vector<int>& distrib_irn, 
  std::vector<int>& distrib_jrn, const int size, const int rank) {
// Calculate distribution sizes
  const int total_elements = nz();
  const int base_size = total_elements / size;
  const int remainder = total_elements % size;

  // Root gets extra elements so that there is minimal communication to distant cores
  const int my_size = (rank == 0) ? base_size + remainder : base_size;

  // Resize destination vectors based on how many elements this process will receive
  if a.size() != my_size {distrib_a.resize(my_size);}
  if irn.size() != my_size {distrib_irn.resize(my_size);}
  if jrn.size() != my_size {distrib_jrn.resize(my_size);}

  // Create arrays for counts and displacements
  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);

  // Set up counts and displacements
  displs[0] = 0;
  sendcounts[0] = base_size + remainder; //core 0 gets the most element

  for (int i = 1; i < size; ++i) {
    displs[i] = displs[i-1] + sendcounts[i-1]; //start sending data to core i+1 located just after the end of data sent to core i
    sendcounts[i] = base_size; // all cores except 0 get the same amount of data
  }

  // Scatter using variable counts
  MPI_Scatterv(a.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
  distrib_a.data(), my_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(irn.data(), sendcounts.data(), displs.data(), MPI_INT,
  distrib_irn.data(), my_size, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(jcn.data(), sendcounts.data(), displs.data(), MPI_INT,
  distrib_jrn.data(), my_size, MPI_INT, 0, MPI_COMM_WORLD);
}
