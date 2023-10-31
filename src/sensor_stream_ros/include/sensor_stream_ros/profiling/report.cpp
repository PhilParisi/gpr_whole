#include "report.h"
namespace ss { namespace profiling {
Report::Report()
{


}

void Report::addSeries(ss::profiling::Series::Ptr series){
  series_.push_back(series);
  index_map_[series->getName()] = series_.size()-1;
}

Series::Ptr &Report::getSeries(size_t index){
  return  series_[index];
}

Series::Ptr &Report::getSeries(std::string key){
  return getSeries(index_map_[key]);
}

void Report::addMetric(ss::profiling::Metric::Ptr metric){
  metrics_.push_back(metric);
  metrics_index_map_[metric->getName()] = series_.size()-1;
}



YAML::Node Report::toYaml(){
  YAML::Node output;
  output["description"] = description_;
  for(Series::Ptr series: getSeriesVect()){
    output["series"][series->getName()] = series->toYaml();
  }
  for(Metric::Ptr metric: metrics_){
    output["metrics"][metric->getName()] = metric->toYaml();
  }
  return output;
}



}}
