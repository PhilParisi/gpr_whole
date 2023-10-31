#include "bpslam_particle_widget.h"
#include "ui_bpslam_particle_widget.h"

BpslamParticleWidget::BpslamParticleWidget(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::BpslamParticleWidget)
{
  ui->setupUi(this);
  setParticle(nullptr);
}

BpslamParticleWidget::~BpslamParticleWidget()
{
  delete ui;
}

void BpslamParticleWidget::setParticle(ss::bpslam::ParticlePtr_t particle_ptr){
  particle_=particle_ptr;
  if(particle_ptr==nullptr){
    setEnabled(false);
  }else {
    setEnabled(true);
    ui->id->setNum(int(particle_->getId()));
    QDateTime myDateTime;
    myDateTime.setTime_t(particle_->getData()->nav.start_time.toSec());
    ui->start_time->setText(myDateTime.toString("yyyy-MM-dd_hh:mm:ss"));
    myDateTime.setTime_t(particle_->getData()->nav.end_time.toSec());
    ui->end_time->setText(myDateTime.toString("yyyy-MM-dd_hh:mm:ss"));
    QString str;
    str.sprintf("x:(%.1f:%.1f) y:(%.1f:%.1f)",
                particle_->getData()->map.prediction_reigon.min_point.x,
                particle_->getData()->map.prediction_reigon.max_point.x,
                particle_->getData()->map.prediction_reigon.min_point.y,
                particle_->getData()->map.prediction_reigon.max_point.y);
    ui->boundary->setText(str);
    ui->likelihood->setNum(particle_->getData()->map.likelihood);
    particle_pub_.publishTrainingBoundary(particle_ptr);
  }
}

void BpslamParticleWidget::publishOdom(){
  particle_pub_.publishOdomHypothesis(particle_);
}

void BpslamParticleWidget::on_hypothesis_btn_clicked()
{
  particle_pub_.publishProjected(particle_);
}

void BpslamParticleWidget::on_gpr_btn_clicked()
{
  particle_pub_.publishGprPrediction(particle_);
  ui->likelihood->setNum(particle_->getData()->map.likelihood);
}


void BpslamParticleWidget::on_gpr_training_btn_clicked()
{
  particle_pub_.PublishGPRTraining(particle_);
}

void BpslamParticleWidget::on_recompute_gpr_clicked()
{
  particle_pub_.publishGprPrediction(particle_,true);
  ui->likelihood->setNum(particle_->getData()->map.likelihood);
}

void BpslamParticleWidget::on_mean_btn_clicked()
{
  particle_pub_.publishMeanFunction(particle_);
  //ui->likelihood->setNum(particle_->getData()->map.likelihood);
}

void BpslamParticleWidget::on_save_map_btn_clicked()
{
  QString fileName = QFileDialog::getSaveFileName(this, tr("Save Particle Trajectory"),
                                                  "/home/untitled.pcd",
                                                  tr("pcd File (*.pcd)"));
  particle_pub_.saveParticleMap(particle_,fileName.toStdString());
}
