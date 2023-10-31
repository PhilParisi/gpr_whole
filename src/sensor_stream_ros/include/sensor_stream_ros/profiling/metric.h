#ifndef METRIC_H
#define METRIC_H

#include <yaml-cpp/yaml.h>
#include <stdexcept>


namespace ss { namespace profiling {

class Metric
{
public:
  typedef  std::shared_ptr<Metric> Ptr;
  typedef  std::shared_ptr<const Metric> ConstPtr;
  Metric();
  virtual YAML::Node toYaml(){throw std::logic_error("to_yaml has not been been implemented for this datatype");}
  std::string getName(){return name_;}
protected:
  void setName(std::string name){name_=name;}
  std::string name_;
};

}}

#endif // METRIC_H
