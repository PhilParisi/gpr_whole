#include "bpslam_widget.h"
#include "ui_bpslam_widget.h"

BPSlamWidget::BPSlamWidget(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::BPSlamWidget)
{
  ui->setupUi(this);


//  ui->error_plot->addGraph();
//  ui->error_plot->graph(0)->setPen(QPen(Qt::red));
//  ui->error_plot->addGraph();
//  ui->error_plot->graph(1)->setPen(QPen(Qt::green));
//  ui->error_plot->addGraph();
//  ui->error_plot->graph(2)->setPen(QPen(Qt::blue));

//  ui->error_plot->addGraph();
//  //ui->error_plot->graph(3)->setName("Confidence Band 68%");
//  ui->error_plot->graph(3)->setPen(QPen(QColor(255,50,30,20)));
//  ui->error_plot->graph(3)->setBrush(QBrush(QColor(255,50,30,20)));
//  ui->error_plot->addGraph();
//  ui->error_plot->legend->removeItem(ui->error_plot->legend->itemCount()-1); // don't show two confidence band graphs in legend
//  ui->error_plot->graph(4)->setPen(QPen(QColor(255,50,30,20)));
//  ui->error_plot->graph(3)->setChannelFillGraph(ui->error_plot->graph(4));

//  ui->error_plot->addGraph();
//  //ui->error_plot->graph(3)->setName("Confidence Band 68%");
//  ui->error_plot->graph(5)->setPen(QPen(QColor(30,255,30,20)));
//  ui->error_plot->graph(5)->setBrush(QBrush(QColor(30,255,30,20)));
//  ui->error_plot->addGraph();
//  ui->error_plot->legend->removeItem(ui->error_plot->legend->itemCount()-1); // don't show two confidence band graphs in legend
//  ui->error_plot->graph(6)->setPen(QPen(QColor(30,255,30,20)));
//  ui->error_plot->graph(5)->setChannelFillGraph(ui->error_plot->graph(6));

  plotter = new SeriesPlotArray;

  setupSlots();


}

BPSlamWidget::~BPSlamWidget()
{
  delete ui;
}
void BPSlamWidget::setBPSlam(ss::bpslam::BPSlam::Ptr bpslam_ptr){
  bpslam_ptr_=bpslam_ptr;

  ui->regression_params->setRegressionParams(bpslam_ptr_->config_.gpr_params.regression);

  QStringList strList;
  for (size_t i = 0;i<ss::idx::cov_index_size; i++) {
    strList << QString::fromStdString(ss::idx::covIndex2String[i]);
  }
  //strList << "X" << "Y" << "Z" << "Roll" << "Pitch" << "Yaw";

  ui->random_vars->addItems(strList);
  QListWidgetItem* item = 0;
      for(int i = 0; i < ui->random_vars->count(); ++i){
          item = ui->random_vars->item(i);
          item->setFlags(item->flags() | Qt::ItemIsUserCheckable);
          item->setCheckState(Qt::Unchecked);
      }
  plotter->setReport(bpslam_ptr->getReportPtr());
  plotter->show();

  updateValues();
  setupSlots();


}

void BPSlamWidget::setupSlots(){
  connect(ui->particle_lifespan,          SIGNAL(valueChanged(double)), this, SLOT(updateValues()));
  connect(ui->particle_n_children,        SIGNAL(valueChanged(int)), this, SLOT(updateValues()));
  connect(ui->max_particles,              SIGNAL(valueChanged(int)), this, SLOT(updateValues()));
  connect(ui->min_particles,              SIGNAL(valueChanged(int)), this, SLOT(updateValues()));
  connect(ui->steps_between_cull,         SIGNAL(valueChanged(int)), this, SLOT(updateValues()));
  connect(ui->min_model_particle_age,     SIGNAL(valueChanged(double)), this, SLOT(updateValues()));
  connect(ui->ekf_uncertainty_multiplier, SIGNAL(valueChanged(double)), this, SLOT(updateValues()));
}

void BPSlamWidget::updateValues(){
  bpslam_ptr_->config_.particle.lifespan = ui->particle_lifespan->value();
  bpslam_ptr_->config_.particle.n_children = ui->particle_n_children->value();
  bpslam_ptr_->config_.max_particles = ui->max_particles->value();
  bpslam_ptr_->config_.min_particles = ui->min_particles->value();
  bpslam_ptr_->config_.steps_between_cull = ui->steps_between_cull->value();
  bpslam_ptr_->config_.min_model_particle_age.fromSec(ui->min_model_particle_age->value());
  bpslam_ptr_->config_.ekf_params->uncertainty_multiplier = ui->ekf_uncertainty_multiplier->value();
}

//void BPSlamWidget::on_particle_lifespan_valueChanged(double arg1)
//{
//    bpslam_ptr_->config_.particle.lifespan = arg1;
//}

//void BPSlamWidget::on_particle_n_children_valueChanged(int arg1)
//{
//    bpslam_ptr_->config_.particle.n_children = arg1;
//}

//void BPSlamWidget::on_max_particles_valueChanged(int arg1)
//{
//    bpslam_ptr_->config_.max_particles = arg1;
//}

void BPSlamWidget::on_random_vars_itemChanged(QListWidgetItem *item)
{
  bpslam_ptr_->config_.ekf_params->random_vars.clear();
  for(size_t i = 0; i < ui->random_vars->count(); ++i){
    if(ui->random_vars->item(i)->checkState()==Qt::Checked)
      bpslam_ptr_->config_.ekf_params->random_vars.push_back(ss::idx::int2CovIdx[i]);
  }
  return;
}


void BPSlamWidget::updateUI(){
  ui->leaf_particles->clear();
  for (auto particle: bpslam_ptr_->getLeafQueue()) {
    ParticleListItem * item = new ParticleListItem;
      item->particle_pointer = particle;
      item->setText(QString::number(particle->getId()));
      ui->leaf_particles->addItem(item);
  }
  std::unordered_set<ss::bpslam::ParticlePtr_t> leaf_queue = bpslam_ptr_->getLeafQueue();
  ui->num_leaf_particles->setNum(int(leaf_queue.size()));
  ss::bpslam::ParticlePtr_t first = *leaf_queue.begin();
  std::list<nav_msgs::Odometry::ConstPtr>::iterator input_odom_it = first->getData()->nav.odom_front;
//  for (size_t odom_idx=0; odom_idx < first->getData()->nav.hypothesis.size() ; odom_idx++) {
//    geometry_msgs::Pose avg_pose;
//    geometry_msgs::Point pos_error;
//    geometry_msgs::Point pos_var;
//    pos_var.x = avg_pose.position.x = 0;
//    pos_var.y = avg_pose.position.y = 0;
//    pos_var.z = avg_pose.position.z = 0;
//    size_t num = 0;
//    double odom_time = first->getData()->nav.hypothesis[odom_idx]->header.stamp.toSec();
//    for(auto particle_prt : leaf_queue){
//      avg_pose.position.x += particle_prt->getData()->nav.hypothesis[odom_idx]->pose.pose.position.x;
//      avg_pose.position.y += particle_prt->getData()->nav.hypothesis[odom_idx]->pose.pose.position.y;
//      avg_pose.position.z += particle_prt->getData()->nav.hypothesis[odom_idx]->pose.pose.position.z;
//      num++;
//    }
//    avg_pose.position.x=avg_pose.position.x/num;
//    avg_pose.position.y=avg_pose.position.y/num;
//    avg_pose.position.z=avg_pose.position.z/num;

//    pos_error.x = input_odom_it->operator->()->pose.pose.position.x - avg_pose.position.x;
//    pos_error.y = input_odom_it->operator->()->pose.pose.position.y - avg_pose.position.y;
//    pos_error.z = input_odom_it->operator->()->pose.pose.position.z - avg_pose.position.z;

//    ui->error_plot->graph(0)->addData(odom_time,pos_error.x);
//    ui->error_plot->graph(1)->addData(odom_time,pos_error.y);
//    ui->error_plot->graph(2)->addData(odom_time,pos_error.z);

//    for(auto particle_prt : leaf_queue){
//      pos_var.x += pow(particle_prt->getData()->nav.hypothesis[odom_idx]->pose.pose.position.x - avg_pose.position.x,2);
//      pos_var.y += pow(particle_prt->getData()->nav.hypothesis[odom_idx]->pose.pose.position.y - avg_pose.position.y,2);
//    }

//    pos_var.x = pos_var.x/num;
//    ui->error_plot->graph(3)->addData(odom_time,pos_error.x+sqrt(pos_var.x));
//    ui->error_plot->graph(4)->addData(odom_time,pos_error.x-sqrt(pos_var.x));

//    pos_var.y = pos_var.y/num;
//    ui->error_plot->graph(5)->addData(odom_time,pos_error.y+sqrt(pos_var.y));
//    ui->error_plot->graph(6)->addData(odom_time,pos_error.y-sqrt(pos_var.y));


//    input_odom_it++;
//  }
//  ui->error_plot->rescaleAxes();
//  ui->error_plot->replot();

  plotter->update();

  uiUpdateComplete();
}


void BPSlamWidget::on_leaf_particles_currentItemChanged(QListWidgetItem *current, QListWidgetItem *previous)
{
  if(current){
    ParticleListItem * particle_item = static_cast<ParticleListItem *>( current);
    ui->particle_details->setParticle(particle_item->particle_pointer);
    ui->particle_details->publishOdom();
  }
}

//void BPSlamWidget::on_steps_between_cull_valueChanged(int arg1)
//{
//    bpslam_ptr_->config_.steps_between_cull = arg1;
//}

void BPSlamWidget::on_pub_particle_tree_clicked()
{
    particle_pub_.publishAncestorOdom(bpslam_ptr_->rootParticlePtr());
}


//void BPSlamWidget::on_min_model_particle_age_valueChanged(double arg1)
//{
//    bpslam_ptr_->config_.min_model_particle_age.fromSec(arg1);
//}

void BPSlamWidget::on_optimize_btn_clicked()
{
  ParticleListItem * particle_item = static_cast<ParticleListItem *>( ui->leaf_particles->currentItem());
  if(particle_item==nullptr){
    return;
  }
  bpslam_ptr_->optimizeGpr(particle_item->particle_pointer);
}

void BPSlamWidget::on_generate_lml_plot_clicked()
{
  // configure axis rect:
  ui->lml_plot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  ui->lml_plot->axisRect()->setupFullAxesBox(true);
  ui->lml_plot->xAxis->setLabel("L");
  ui->lml_plot->yAxis->setLabel("Sigma");



  // set up the QCPColorMap:
//  if(colorMap)
//    delete colorMap;
  colorMap = new QCPColorMap(ui->lml_plot->xAxis, ui->lml_plot->yAxis);
  int nx = 10;
  int ny = 10;
  colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
  colorMap->data()->setRange(QCPRange(.25, 1.25), QCPRange(-1.5, -.25)); // and span the coordinate range -4..4 in both key (x) and value (y) dimensions
  // now we assign some data, by accessing the QCPColorMapData instance of the color map:









  // add a color scale:
  QCPColorScale *colorScale = new QCPColorScale(ui->lml_plot);
  ui->lml_plot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
  colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
  colorMap->setColorScale(colorScale); // associate the color map with the color scale
  colorScale->axis()->setLabel("LML");

  // set the color gradient of the color map to one of the presets:
  colorMap->setGradient(QCPColorGradient::gpJet);
  // we could have also created a QCPColorGradient instance and added own colors to
  // the gradient, see the documentation of QCPColorGradient for what's possible.

  // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:


  // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
  QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->lml_plot);
  ui->lml_plot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
  colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);






  max_lml_=-INFINITY;
  double x, y, z;
  for (int xIndex=0; xIndex<nx; ++xIndex)
  {
    for (int yIndex=0; yIndex<ny; ++yIndex)
    {
      colorMap->data()->cellToCoord(xIndex, yIndex, &x, &y);
//      double r = 3*qSqrt(x*x+y*y)+1e-2;
//      z = 2*x*(qCos(r+2)/r-qSin(r+2)/r); // the B field strength of dipole radiation (modulo physical constants)
//      colorMap->data()->setCell(xIndex, yIndex, z);
      ParticleListItem * particle_item = static_cast<ParticleListItem *>( ui->leaf_particles->currentItem());
      if(particle_item==nullptr){
        return;
      }

      bpslam_ptr_->config_.gpr_params.regression->kernel->hyperparam(0) = pow(10,x);
      bpslam_ptr_->config_.gpr_params.regression->kernel->hyperparam(1) = pow(10,y);
      bpslam_ptr_->config_.gpr_params.regression->kernel->hyperparam2dev();
//      params.reset();
//      *params.regression = *bpslam_ptr_->config_.gpr_params.regression;
//      params.regression->
      ss::bpslam::GPRWorker worker(particle_item->particle_pointer,bpslam_ptr_->config_.gpr_params);

      worker.setupGPR();
      //worker.computeMeanFunction(particle_item->particle_pointer->getParent(),100 ,true);
      worker.addBlocks2GPR(particle_item->particle_pointer);

      z = particle_item->particle_pointer->getData()->map.gpr->lml();
      if(z>max_lml_){
        max_lml_=z;
      }
      std::cout << "x,y,z: " << x << " " << y << " " << z << std::endl;
      colorMap->data()->setCell(xIndex, yIndex, z);

      colorMap->rescaleDataRange();
      ui->lml_plot->rescaleAxes();
      ui->lml_plot->replot();
    }
  }



  // rescale the key (x) and value (y) axes so the whole color map is visible:

}

void BPSlamWidget::on_min_lml_valueChanged(double arg1)
{
  if(colorMap){
    QCPRange range(arg1,max_lml_);
    colorMap->setDataRange(range);
    ui->lml_plot->rescaleAxes();
    ui->lml_plot->replot();
  }
}

void BPSlamWidget::on_plotter_btn_clicked()
{
  plotter->show();
}


void BPSlamWidget::on_save_metric_btn_clicked()
{
  QString fileName = QFileDialog::getSaveFileName(this, tr("Specify A URDF File"),
                                                  "/home",
                                                  tr("YAML File (*.yaml)"));
  if(fileName.length()>0){
    bpslam_ptr_->saveMetrics(fileName.toStdString());
  }
}

void BPSlamWidget::on_metrics_file_btn_clicked()
{
  QString fileName = QFileDialog::getSaveFileName(this, tr("Specify A URDF File"),
                                                  "/home",
                                                  tr("YAML File (*.yaml)"));
  if(fileName.length()>0){
    ui->metrics_file->setText(fileName);
  }
}

void BPSlamWidget::on_metrics_file_textChanged(const QString &arg1)
{
  bpslam_ptr_->setMetricsFile(arg1.toStdString());
}
