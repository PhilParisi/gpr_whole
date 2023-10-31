#ifndef REPORT_H
#define REPORT_H

#include "series.h"
#include "yaml_metric.h"
#include <memory>

namespace ss { namespace profiling {
/*!
 * \brief A container for all the profiling metrics available
 * this class can be thought of as an analog for files you want to
 * store on disk or display in plots
 */
class Report
{
public:
  typedef std::shared_ptr<Report> Ptr;

  Report();

  /*!
   * \brief Add a series to the report
   * \param series a pointer to the series you want to add
   */
  void addSeries(ss::profiling::Series::Ptr series);

  /*!
   * \brief getSeries get a series pointer by index
   * \param index
   * \return
   */
  ss::profiling::Series::Ptr & getSeries(size_t index);

  /*!
   * \brief getSeries get a series pointer by key
   * \param index
   * \return
   */
  ss::profiling::Series::Ptr & getSeries(std::string key);

  void addMetric(ss::profiling::Metric::Ptr metric);
  ss::profiling::Metric::Ptr & getMetric(size_t index);
  ss::profiling::Metric::Ptr & getMetric(std::string key);


  YAML::Node toYaml();

  std::vector<ss::profiling::Series::Ptr> getSeriesVect(){return series_;}
  void setDescription(std::string desc){description_=desc;}
protected:
  std::vector<ss::profiling::Series::Ptr> series_;  ///< the storage for the series
  std::map<std::string, size_t> index_map_;         ///< in case you want to access the series by key instead
  std::vector<ss::profiling::Metric::Ptr> metrics_;  ///< the storage for the series
  std::map<std::string, size_t> metrics_index_map_;         ///< in case you want to access the series by key instead
  std::string description_;
};

}}

#endif // REPORT_H
