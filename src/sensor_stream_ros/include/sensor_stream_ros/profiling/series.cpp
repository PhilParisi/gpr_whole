#include "series.h"
namespace ss { namespace profiling {

Series::Series(std::string name)
{
  setName(name);
}

YAML::Node Series::toYaml(){
  YAML::Node output;
  for (const auto& kv : float_data_) {
    output["float_data"][kv.first] = float_data_[kv.first];
  }
  for (const auto& kv : int_data_) {
    output["int_data"][kv.first] = int_data_[kv.first];
  }
  for (const auto& kv : variance_) {
    output["variance"][kv.first] = variance_[kv.first];
  }
  return output;
}


bool Series::hasValue(std::string key){
  bool out = false;
  if(float_data_.count(key)>0){
    out = true;
  }
  if(int_data_.count(key)>0){
    out = true;
  }
  return out;
}

bool Series::hasVariance(std::string key){
  return variance_.count(key)>0;
}


double Series::getValue(std::string key, size_t index){
  double value = NAN;
  if(float_data_.count(key)>0){
    value = float_data_[key][index];
  }
  if(int_data_.count(key)>0){
    value = int_data_[key][index];
  }
  return value;
}

double Series::getVariance(std::string key, size_t index){
  double value = NAN;
  if(variance_.count(key)>0){
    value = variance_[key][index];
  }
  return value;
}

std::vector<std::string> Series::getKeys(){
  std::vector<std::string> keys;
  for (const auto& kv : float_data_) {
    keys.push_back(kv.first);
  }
  for (const auto& kv : int_data_) {
    keys.push_back(kv.first);
  }

  return  keys;
}

size_t Series::size(){
  size_t size=0;
  for (const auto& kv : float_data_) {
    size = std::max(kv.second.size(),size);
  }
  for (const auto& kv : int_data_) {
    size = std::max(kv.second.size(),size);
  }
  return size;
}

}}


