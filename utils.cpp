#include<algorithm>
#include<numeric>
#include <iterator>
#include<vector>


template<typename T>
std::vector<std::size_t> argsort(const std::vector<T>& row){
    const int n = row.size();
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&row](std::size_t i, std::size_t j){return row[i]<row[j];});
    return indices;
}

template<typename T>
std::vector<T> reorderVector(const std::vector<T>& my_vec, const std::vector<std::size_t>& indices) {
    std::vector<T> reordered_vec(my_vec.size());
    std::transform(indices.begin(), indices.end(), reordered_vec.begin(), [&my_vec](std::size_t idx) {
        return my_vec[idx];
    });
    return reordered_vec;
}