#ifndef BPSLAM_WIDGET_H
#define BPSLAM_WIDGET_H

#include <QWidget>
#include <QListWidgetItem>
#include <sensor_stream_ros/bpslam/bpslam.h>
#include <sensor_stream_ros/ui/bpslam_particle_widget.h>
#include <sensor_stream_ros/third_party/qcustomplot.h>
#include "profiling/series_plot_array.h"

namespace Ui {
class BPSlamWidget;
}

class ParticleListItem : public QListWidgetItem{
public:
  ss::bpslam::ParticlePtr_t particle_pointer;
};

class BPSlamWidget : public QWidget
{
  Q_OBJECT

public:
  explicit BPSlamWidget(QWidget *parent = nullptr);
  ~BPSlamWidget();
  void setBPSlam(ss::bpslam::BPSlam::Ptr bpslam_ptr);
  void setupSlots();

signals:
  void uiUpdateComplete();

public slots:
  void updateUI();

private slots:
  void updateValues();
  void on_random_vars_itemChanged(QListWidgetItem *item);
  void on_leaf_particles_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous);
  void on_pub_particle_tree_clicked();

  void on_optimize_btn_clicked();

  void on_generate_lml_plot_clicked();

  void on_plotter_btn_clicked();


  void on_min_lml_valueChanged(double arg1);

  void on_save_metric_btn_clicked();

  void on_metrics_file_btn_clicked();

  void on_metrics_file_textChanged(const QString &arg1);

private:
  Ui::BPSlamWidget *ui;
  ss::bpslam::BPSlam::Ptr bpslam_ptr_;
  ss::bpslam::ParticlePublisher particle_pub_;
  QCPColorMap *colorMap ;
  SeriesPlotArray *plotter;
  float max_lml_;
};

#endif // BPSLAM_WIDGET_H
