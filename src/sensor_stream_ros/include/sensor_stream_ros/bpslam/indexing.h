#ifndef INDEXING_H
#define INDEXING_H
#include <cstddef>
#include <string>
namespace ss {namespace idx {


enum covIndex {x_linear,y_linear,z_linear,x_angular,y_angular,z_angular};
const size_t cov_index_size = 6;
const std::string covIndex2String[6] = {"X Linear","Y Linear","Z Linear","X Angular","Y Angular","Z Angular"};
const covIndex int2CovIdx[6] = {x_linear,y_linear,z_linear,x_angular,y_angular,z_angular};
inline size_t rowMajor(size_t row, size_t col, size_t cols){
  return row*cols+col;
}

}}



#endif // INDEXING_H
