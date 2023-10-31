#ifndef BPSLAM_BAG_WORKER_H
#define BPSLAM_BAG_WORKER_H

#include <QObject>
#include <QTimer>
#include <sensor_stream_ros/bpslam/bpslam_bag_processor.h>
#include <QMessageBox>
#include <QtCore/QThread>

enum RunMode {step_once, step_continuous, paused};

class BPSLAMBagWorker : public QObject
{
  Q_OBJECT
public:
  BPSLAMBagWorker(ss::bpslam::BagProcessor::Ptr bag_processor_ptr);

signals:
  void ready2spawn();

public slots:
  void start();
  void onGuiUpdate();
  void pause();
  void stepOnce();
  void cullParticles();
  void computeLeafsGPR();
  //void stop();

private slots:
    void iterate();

private:
  ss::bpslam::BagProcessor::Ptr bag_processor_ptr_;
  QTimer * timer_;
  RunMode run_mode_;

};

#endif // BPSLAM_BAG_WORKER_H
