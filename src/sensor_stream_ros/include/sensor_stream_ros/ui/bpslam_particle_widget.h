#ifndef BPSLAM_PARTICLE_WIDGET_H
#define BPSLAM_PARTICLE_WIDGET_H

#include <QWidget>
#include <QDateTime>
#include <sensor_stream_ros/bpslam/bpslam.h>
#include <sensor_stream_ros/bpslam/particle_publisher.h>
#include <qfiledialog.h>

namespace Ui {
class BpslamParticleWidget;
}

class BpslamParticleWidget : public QWidget
{
  Q_OBJECT


public:
  explicit BpslamParticleWidget(QWidget *parent = nullptr);
  ~BpslamParticleWidget();
  void setParticle(ss::bpslam::ParticlePtr_t particle_ptr);
  void publishOdom();

signals:
  void pubHypothesisClicked();

private slots:
  void on_hypothesis_btn_clicked();
  void on_gpr_btn_clicked();
  
  void on_gpr_training_btn_clicked();

  void on_recompute_gpr_clicked();

  void on_mean_btn_clicked();

  void on_save_map_btn_clicked();

private:
  Ui::BpslamParticleWidget *ui;
  ss::bpslam::ParticlePtr_t particle_;
  ss::bpslam::ParticlePublisher particle_pub_;
};

#endif // BPSLAM_PARTICLE_WIDGET_H
