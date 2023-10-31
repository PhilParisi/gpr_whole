#ifndef SERIES_PLOT_ARRAY_H
#define SERIES_PLOT_ARRAY_H

#include <QWidget>
#include "series_plot_editor.h"
#include "include/sensor_stream_ros/profiling/report.h"

namespace Ui {
class SeriesPlotArray;
}

class SeriesPlotArray : public QWidget
{
  Q_OBJECT

public:
  explicit SeriesPlotArray(QWidget *parent = nullptr);
  ~SeriesPlotArray();
  /*!
   * \brief Add a plot to the plotarray
   * \param index the index of the series you want to add
   */
  void addPlot(size_t index);
  /*!
   * \brief Add a series to plottable series
   * \param series the series you want to add
   */
  void setReport(ss::profiling::Report::Ptr report);

public slots:
  void update();

private slots:
  void on_add_plot_clicked();

private:
  Ui::SeriesPlotArray *ui;
  std::vector<SeriesPlotEditor *> plots_;
  //std::vector<ss::profiling::Series::Ptr> series_;
  ss::profiling::Report::Ptr report_;
};

#endif // SERIES_PLOT_ARRAY_H
