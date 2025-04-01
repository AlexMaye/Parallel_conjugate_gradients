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
  size_t k = 0;
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

void CGSolverSparse::parallel_solve(std::vector<double>& x, const std::vector<double>& m_b, const int m_n, const int size, const int rank){
  //A is a m_m x m_n matrix
  
  std::vector<double> r(m_n);
  std::vector<double> p(m_n);
  std::vector<double> local_Ap(m_n);
  std::vector<double> tmp(m_n);

  const double my_tolerance = 1e-10;

  int total_elements;
  if rank == 0
    total_elements = m_A.nz();
  
  MPI_Bcast(&total_elements, 1, MPI_INT, 0, MPI_COMM_WORLD);
  const int base_size = total_elements / size;
  const int remainder = total_elements % size;
  const int distrib_size = (rank == 0) ? base_size + remainder : base_size;
  std::vector<double> distrib_a(distrib_size, 0);
  std::vector<int> distrib_irn(distrib_size, 0);
  std::vector<int> distrib_jcn(distrib_size, 0);

  m_A.distribute_mat(distrib_a, distrib_irn, distrib_jcn, size, rank); //scatter the values of m_A to other processes.
  MatrixCOO distrib_m_A(distrib_a, distrib_irn, distrib_jcn);

  // r = b - A * x;
  

  distrib_m_A.mat_vec(x, local_Ap); 

  std::vector<double> Ap(m_n);
  MPI_AllReduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

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
  size_t k = 0;
  for (; k < m_n; ++k)
  {
    // Ap = A * p;
    distrib_m_A.mat_vec(p, local_Ap);
    MPI_AllReduce(local_Ap.data(), Ap.data(), m_n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


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
    if (std::sqrt(rsnew) < my_tolerance)
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
