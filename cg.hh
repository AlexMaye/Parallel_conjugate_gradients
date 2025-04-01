#include "matrix_coo.hh"
#include <cblas.h>
#include <string>
#include <vector>

#ifndef __CG_HH__
#define __CG_HH__

/*v
 * oid CGSolver::solve(std::vector<double> & x)
void CGSolver::read_matrix(const std::string & filename)
void CGSolverSparse::solve(std::vector<double> & x)
void CGSolverSparse::read_matrix(const std::string & filename)
void MatrixCSR::mvm(const std::vector<double> & x, std::vector<double> & y)
const void MatrixCSR::loadMMMatrix(const std::string & filename) void
Solver::init_source_term(int n, double h)
*/
class Solver
{
public:
  virtual void read_matrix(const std::string &filename) = 0;
  void init_source_term(const double h);
  virtual void solve(std::vector<double> &x) = 0;

  inline int m() const
  {
    return m_m;
  }
  inline int n() const
  {
    return m_n;
  }

  void tolerance(double tolerance)
  {
    m_tolerance = tolerance;
  }

  void set_rows_cols(const int new_m_m, const int new_m_n){
    m_m = new_m_m;
    m_n = new_m_n;
  }

  std::vector<double> get_m_b() const {return m_b;}

private:
  int m_m{0}; //number of rows
  int m_n{0}; //number of cols
  std::vector<double> m_b; //rhs of Ax = b
  double m_tolerance{1e-10};
};

class CGSolverSparse : public Solver
{
public:
  CGSolverSparse() = default;
  virtual void read_matrix(const std::string &filename);
  virtual void solve(std::vector<double> &x);
  void parallel_solve(std::vector<double>& x, const std::vector<double>& m_b, const int m_n, const int size, const int rank);

private:
  MatrixCOO m_A;
};

#endif /* __CG_HH__ */
