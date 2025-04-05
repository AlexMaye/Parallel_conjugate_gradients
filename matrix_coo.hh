#include <algorithm>
#include <string>
#include <vector>

#ifndef __MATRIX_COO_H_
#define __MATRIX_COO_H_

class MatrixCOO {
public:
  MatrixCOO() = default;
  MatrixCOO(const std::vector<double>& values, const std::vector<int>& row_indices, const std::vector<int>& col_indices);
    

  inline int m() const { return m_m; }
  inline int n() const { return m_n; }

  inline int nz() const { return irn.size(); }
  inline int is_sym() const { return m_is_sym; }

  void read(const std::string & filename);

  void mat_vec(const std::vector<double> & x, std::vector<double> & y) {
    std::fill_n(y.begin(), y.size(), 0.);

    for (std::size_t z = 0; z < irn.size(); ++z) {
      const auto i = irn[z];
      const auto j = jcn[z];
      const auto a_ = a[z];

      y[i] += a_ * x[j];
      if (m_is_sym and (i != j)) {
        y[j] += a_ * x[i];
      }
    }
  }

  void distribute_mat(std::vector<double>& a, std::vector<int>& distrib_irn, std::vector<int>& distrib_jrn, const int size, const int rank);

  std::vector<int> irn;
  std::vector<int> jcn;
  std::vector<double> a;

private:
  int m_m{0};
  int m_n{0};
  bool m_is_sym{false};
  void get_displs_sdcts(const std::vector<int>& irn, const int size, std::vector<int>& displs, std::vector<int>& sendcounts);
};

#endif // __MATRIX_COO_H_
