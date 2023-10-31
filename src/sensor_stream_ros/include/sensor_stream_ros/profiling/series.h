#ifndef SERIES_H
#define SERIES_H

#include "metric.h"
#include <map>
#include <string>
#include <memory>
#include <cmath>

namespace ss { namespace profiling {

/*!
 * \brief a double value with associted uncertainty
 */
struct UncertainValue : public Metric{
  double value;
  double variance;
};

/*!
 * \brief The Series class is used to represent some set of data relative to
 * an index.  Think Timeseries
 */
class Series : public Metric
{
public:
  typedef  std::shared_ptr<Series> Ptr;
  typedef  std::shared_ptr<const Series> ConstPtr;

  Series(std::string name);

  YAML::Node toYaml();

  void pushBack(std::string key, double val){float_data_[key].push_back(val);}
  void pushBack(std::string key, int val){int_data_[key].push_back(val);}
  void pushBack(std::string key, size_t val){int_data_[key].push_back(val);}
  void pushBack(std::string key, long int val){int_data_[key].push_back(val);}
  void pushBack(std::string key, double val, double var){float_data_[key].push_back(val); variance_[key].push_back(var);}

  void set(   std::string key, std::vector<double>    vec){float_data_[key] = vec;}
  void set(   std::string key, std::vector<int>  vec){int_data_[key]        = vec;}
  void setVar(std::string key, std::vector<double>    vec){variance_[key]   = vec;}

  const std::vector<double> & float_data(std::string key){return float_data_[key];}
  const std::vector<int>    & int_data(std::string key){return int_data_[key];}
  const std::vector<double> & variance(std::string key){return variance_[key];}

  bool hasValue(std::string key);
  bool hasVariance(std::string key);

  double getValue(std::string key, size_t index);
  double getVariance(std::string key, size_t index);

  std::vector<std::string> getKeys();

  size_t size();



private:
  std::map<std::string, std::vector<double>>   float_data_;
  std::map<std::string, std::vector<int>> int_data_;
  std::map<std::string, std::vector<double>>   variance_;

};

}}
#endif // SERIES_H
