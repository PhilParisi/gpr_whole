#ifndef BPSLAM_BAG_WIDGET_H
#define BPSLAM_BAG_WIDGET_H

#include <QWidget>
#include <QFileDialog>
#include <QMessageBox>
#include <boost/exception/diagnostic_information.hpp>
#include <sensor_stream_ros/bpslam/bpslam_bag_processor.h>
#include "bpslam_bag_worker.h"
namespace Ui {
class BPSlamBagWidget;
}

class BPSlamBagWidget : public QWidget
{
  Q_OBJECT

public:
  explicit BPSlamBagWidget(QWidget *parent = nullptr);
  ~BPSlamBagWidget();
  void setupSignals();
  void setBagProcessor(ss::bpslam::BagProcessor::Ptr bag_processor_ptr);
  void setBagFile(QString bag_file);
  void setUrdfFile(QString urdf_file);
private slots:
  void on_urdf_button_clicked();
  void on_bag_button_clicked();
  void checkVaildFile();
  void on_run_btn_clicked();


  void on_start_btn_clicked();

private:
  Ui::BPSlamBagWidget *ui;
  ss::bpslam::BagProcessor::Ptr bag_processor_ptr_;
  BPSLAMBagWorker * worker_;
  QThread * worker_thread_;
};

#endif // BPSLAM_BAG_WIDGET_H
