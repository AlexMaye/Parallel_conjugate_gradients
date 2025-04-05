#include<algorithm>
#include<numeric>
#include <iterator>
#include<vector>


template<typename T>
std::vector<std::size_t> argsort(const std::vector<T>& row){
    std::vector<std::size_t> indices(row.size());
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

template<typename T, typename S>
void expand_indices(std::vector<T>& irn, std::vector<T>& jcn, std::vector<S>& a){
    
    const std::size_t& original_size = irn.size();
    irn.reserve(2 * original_size);
    jcn.reserve(2 * original_size);
    a.reserve(2 * original_size);
    for (std::size_t z = 0; z < original_size; ++z) {    
        T i = irn[z];
        T j = jcn[z];    
        // Skip diagonal elements (they don't need mirroring)
        if (i != j) {
            irn.push_back(j);  // Note the swap of i and j
            jcn.push_back(i);
            a.push_back(a[z]);
        }
    }
}