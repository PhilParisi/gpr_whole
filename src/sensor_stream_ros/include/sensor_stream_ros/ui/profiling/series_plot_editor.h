#ifndef SERIES_PLOT_EDITOR_H
#define SERIES_PLOT_EDITOR_H

#include <QWidget>
#include "series_plot.h"
#include <algorithm>

namespace Ui {
class SeriesPlotEditor;
}

class SeriesPlotEditor : public QWidget
{
  Q_OBJECT

public:
  explicit SeriesPlotEditor(QWidget *parent = nullptr);
  ~SeriesPlotEditor();
  void setSeries(ss::profiling::Series::Ptr series);

public  slots:
  void update();
  void selectionChanged();



private slots:
  void on_add_plot_clicked();

private:
  Ui::SeriesPlotEditor *ui;
  ss::profiling::Series::Ptr series_;
};

#endif // SERIES_PLOT_EDITOR_H
