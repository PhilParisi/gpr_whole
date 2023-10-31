#ifndef PATCH_TESTER_GUI_H
#define PATCH_TESTER_GUI_H

#include <QWidget>
#include <QDialog>
#include <QMessageBox>
#include <QFileDialog>
#include <qthread.h>
#include <QTimer>
#include <thread>
#include <ros/ros.h>
#include "patchtester.h"
#include <pcl_conversions/pcl_conversions.h>
#include <tf/transform_broadcaster.h>
#include "ping_range_edit.h"
namespace Ui {
class PatchTesterGUI;
}

class PatchTesterGUI : public QWidget
{
  Q_OBJECT

public:
  explicit PatchTesterGUI(QWidget *parent = 0);
  ~PatchTesterGUI();
  void setupConnections();
  void pushBackRange();
  bool canProject(int index);
  sensor_msgs::PointCloud2::Ptr getPreviewRangeMsg(int index);
  void previewRange(int index);

private slots:
  void on_save_btn_clicked();
  void on_range_edit(int index);

private slots:
  void on_project_btn_clicked();
  void on_urdf_btn_clicked();
  void on_bagfile_btn_clicked();
  void progressCallback();
  void on_frame_selector_currentIndexChanged(const QString &arg1);
  void on_roll_valueChanged(double arg1);
  void on_pitch_valueChanged(double arg1);
  void on_yaw_valueChanged(double arg1);
  void on_x_off_valueChanged(double arg1);
  void on_y_off_valueChanged(double arg1);
  void on_z_off_valueChanged(double arg1);
  void on_spinBox_valueChanged(int arg1);
//  void on_pingNumber_valueChanged(int arg1);
//  void on_pingSlider_sliderMoved(int position);
//  void on_pingRange_valueChanged(int arg1);
//  void on_comparePing_valueChanged(int arg1);
  void on_save_trajectory_clicked();
  void on_read_points_btn_clicked();
  void on_map_frame_currentTextChanged(const QString &arg1);

  void on_add_range_btn_clicked();

  void on_preview_ranges_btn_clicked();

protected:
  void project();
  void load();
  void readPoints();
  void setJoint();
  void updateCurrentPing();
  void updateComparePing();
  std::vector<std::thread> _runThreads;
  Ui::PatchTesterGUI *ui;
  PatchTester patchTester_;
  QTimer *progressTimer_;
  ros::NodeHandlePtr node_;
  ros::Publisher cloudPub_;
  ros::Publisher currentPingPub_;
  ros::Publisher comparePingPub_;
  tf::TransformBroadcaster tfBroadcaster_;
  std::vector<PingRangeEdit*> ranges_;
  size_t last_adjusted_index_;
};

#endif // PATCH_TESTER_GUI_H
