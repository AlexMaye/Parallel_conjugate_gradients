#include "cg.hh"
#include <chrono>
#include <iostream>

using clk = std::chrono::high_resolution_clock;
using second = std::chrono::duration<double>;
using time_point = std::chrono::time_point<clk>;

/*
Implementation of a simple CG solver using matrix in the mtx format (Matrix
market) Any matrix in that format can be used to test the code
*/
int sequential_main(int argc, char **argv){
  // Parallel implementation is the function below this one. Keeping this one for reference.
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]" << std::endl;
    return 1;
  }

  CGSolverSparse sparse_solver;
  sparse_solver.read_matrix(argv[1]);
  int n = sparse_solver.n();
  int m = sparse_solver.m();
  double h = 1. / n;

  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  std::cout << "Call CG sparse on matrix size " << m << " x " << n << ")" << std::endl;
  auto t1 = clk::now();
  sparse_solver.solve(x_s);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (sparse solver)  = " << elapsed.count() << " [s]\n";

  return 0;
}

int main(int argc, char **argv){
  // Parallel implementation using MPI.
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " [martix-market-filename]" << std::endl;
    return 1;
  }

  MPI_Init(&argc, &argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  CGSolverSparse sparse_solver;

  int n, m; //prepare variables for size of matrix

  if (rank == 0){
    sparse_solver.read_matrix(argv[1]); //full matrix is only loaded on root core
    n = sparse_solver.n();
    m = sparse_solver.m();
  }

  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);

  const double h = 1. / n;
  sparse_solver.init_source_term(h);

  std::vector<double> x_s(n);
  std::fill(x_s.begin(), x_s.end(), 0.);

  std::cout <<"rank " << rank << " out of " << size << "calls CG sparse on matrix size " << m << " x " << n << ")" << std::endl;
  auto t1 = clk::now();
  
  sparse_solver.parallel_solve(x_s, n, size, rank);
  second elapsed = clk::now() - t1;
  std::cout << "Time for CG (sparse parallel solver) on rank " << rank <<" = " << elapsed.count() << " [s]\n";

  MPI_Finalize();

  return 0;
}
