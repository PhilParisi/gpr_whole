#ifndef YAML_METRIC_H
#define YAML_METRIC_H
#include "metric.h"
#include <memory>
namespace ss { namespace profiling {
class YamlMetric: public Metric
{
public:
  typedef  std::shared_ptr<YamlMetric> Ptr;
  typedef  std::shared_ptr<const YamlMetric> ConstPtr;
  YamlMetric(std::string name);
  YAML::Node toYaml(){return data;}
  YAML::Node data;
};
}}
#endif // YAML_METRIC_H
