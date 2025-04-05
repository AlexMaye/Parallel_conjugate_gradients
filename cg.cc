#include "cg.hh"

#include <algorithm>
#include <cblas.h>
#include <cmath>
#include <iostream>
#include <mpi.h>

const double NEARZERO = 1.0e-14;
const bool DEBUG = false;

/*
    cgsolver solves the linear equation A*x = b where A is
    of size m x n

Code based on MATLAB code (from wikipedia ;-)  ):

function x = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;

    for i = 1:length(b)
        Ap = A * p;
        alpha = rsold / (p' * Ap);
        x = x + alpha * p;
        r = r - alpha * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
              break;
        end
        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end

*/

/*
Sparse version of the cg solver
*/
void
CGSolverSparse::solve(std::vector<double> &x)
{
  //A is a m_m x m_n matrix
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> Ap(m_n);
  std::vector<double> tmp(m_n);

  // r = b - A * x;
  m_A.mat_vec(x, Ap); //result is stored in Ap, i.e Ap = A@x
  r = m_b;
  //daxpy: y = \alpha x + y
  // void cblas_daxpy(const int n, const double alpha, 
  // const double *x, const int incx, 
  // double *y, const int incy);
  // where n is number of elements in vectors, x pointer to source vector, y pointer to destination vector
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;

  // rsold = r' * r;
  double rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    m_A.mat_vec(p, Ap);

    // alpha = rsold / (p' * Ap);
    double alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1), rsold * NEARZERO);

    // x = x + alpha * p;
    cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

    // r = r - alpha * Ap;
    cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

    // rsnew = r' * r;
    double rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (std::sqrt(rsnew) < m_tolerance)
      break; // Convergence test

    double beta = rsnew / rsold;
    // p = r + (rsnew / rsold) * p;
    tmp = r;
    cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
    p = tmp;

    // rsold = rsnew;
    rsold = rsnew;
    if (DEBUG)
    {
      std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
    }
  }

  if (DEBUG)
  {
    m_A.mat_vec(x, r);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res =
      std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}

void CGSolverSparse::parallel_solve(std::vector<double>& x, const int m_n, const int size, const int rank){
  //A is a m_m x m_n matrix
  
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> local_Ap(m_n);
  std::vector<double> Ap(m_n);
  std::vector<double> tmp(m_n);

  std::vector<double> distrib_a;
  std::vector<int> distrib_irn;
  std::vector<int> distrib_jcn;

  bool convergence;


  m_A.distribute_mat(distrib_a, distrib_irn, distrib_jcn, size, rank); //scatter the values of m_A to other processes.
  MatrixCOO distrib_m_A(distrib_a, distrib_irn, distrib_jcn); //create a new sparse COO matrix on each process with the scattered variables

  // r = b - A * x;

  distrib_m_A.mat_vec(x, local_Ap); //Result is stored in a local_Ap of size m_n

  MPI_Allreduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  //The line could be done with an Allgatherv, but here we do not need to bother with counts, displacements and if the matrix A has an empty line.

  r = m_b;
  //daxpy: y = \alpha x + y
  // void cblas_daxpy(const int n, const double alpha, 
  // const double *x, const int incx, 
  // double *y, const int incy);
  // where n is number of elements in vectors, x pointer to source vector, y pointer to destination vector
  cblas_daxpy(m_n, -1., Ap.data(), 1, r.data(), 1);

  // p = r;
  p = r;
  //All processes have p_0

  // rsold = r' * r;
  double rsold = cblas_ddot(m_n, r.data(), 1, r.data(), 1);
  double rsnew;

  // for i = 1:length(b)
  int k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    distrib_m_A.mat_vec(p, local_Ap); //store the result in local_Ap
    MPI_Reduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    //Only rank 0 holds A*p_k = Ap

    if (rank == 0){

      // alpha = rsold / (p' * Ap);
      double alpha = rsold / std::max(cblas_ddot(m_n, p.data(), 1, Ap.data(), 1), rsold * NEARZERO);

      // x = x + alpha * p;
      cblas_daxpy(m_n, alpha, p.data(), 1, x.data(), 1);

      // r = r - alpha * Ap;
      cblas_daxpy(m_n, -alpha, Ap.data(), 1, r.data(), 1);

      // rsnew = r' * r;
      rsnew = cblas_ddot(m_n, r.data(), 1, r.data(), 1);

      convergence = std::sqrt(rsnew) < m_tolerance;
    }

    MPI_Bcast(&convergence, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    // if sqrt(rsnew) < 1e-10
    //   break;
    if (convergence)
      break; // Convergence test
    
    if (rank == 0){

      double beta = rsnew / rsold;
      // p = r + (rsnew / rsold) * p;
      tmp = r;
      cblas_daxpy(m_n, beta, p.data(), 1, tmp.data(), 1);
      p = tmp;

      // rsold = rsnew;
      rsold = rsnew;
      if (DEBUG)
      {
        std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << "\r" << std::flush;
      }
    }
    MPI_Bcast(p.data(), m_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  }
    

  if (DEBUG)
  {
    m_A.mat_vec(x, r);
    cblas_daxpy(m_n, -1., m_b.data(), 1, r.data(), 1);
    auto res =
      std::sqrt(cblas_ddot(m_n, r.data(), 1, r.data(), 1)) / std::sqrt(cblas_ddot(m_n, m_b.data(), 1, m_b.data(), 1));
    auto nx = std::sqrt(cblas_ddot(m_n, x.data(), 1, x.data(), 1));
    std::cout << "\t[STEP " << k << "] residual = " << std::scientific << std::sqrt(rsold) << ", ||x|| = " << nx
              << ", ||Ax - b||/||b|| = " << res << std::endl;
  }
}


void
CGSolverSparse::read_matrix(const std::string &filename)
{
  m_A.read(filename);
  m_m = m_A.m();
  m_n = m_A.n();
}

/*
Initialization of the source term b
*/
void
Solver::init_source_term(const double h)
{
  m_b.resize(m_n);

  for (int i = 0; i < m_n; i++)
  {
    m_b[i] = -2. * i * M_PI * M_PI * std::sin(10. * M_PI * i * h) * std::sin(10. * M_PI * i * h);
  }
}
