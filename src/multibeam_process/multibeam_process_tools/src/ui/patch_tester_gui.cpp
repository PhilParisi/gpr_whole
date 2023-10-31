#include "ui/patch_tester_gui.h"
//#include "/home/kris/catkin_ws/build/multibeam_process/ui_patch_tester_gui.h"
#include "ui_patch_tester_gui.h"

PatchTesterGUI::PatchTesterGUI(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::PatchTesterGUI)
{
  node_.reset( new(ros::NodeHandle) );
  cloudPub_=node_->advertise<sensor_msgs::PointCloud2>("projected_cloud",1);
  currentPingPub_=node_->advertise<sensor_msgs::PointCloud2>("current_ping_projected",1);
  comparePingPub_=node_->advertise<sensor_msgs::PointCloud2>("compare_ping_projected",1);
  ui->setupUi(this);
  progressTimer_ = new QTimer(this);
  connect(progressTimer_, SIGNAL(timeout()), this, SLOT(progressCallback()));
  progressTimer_->start(500);
  ui->project_btn->setEnabled( false );

  pushBackRange();  // set initial range
}

PatchTesterGUI::~PatchTesterGUI()
{
  for (auto& th : _runThreads) th.join();
  delete ui;
}

void PatchTesterGUI::setupConnections(){

}

void PatchTesterGUI::pushBackRange(){
  ranges_.push_back(new PingRangeEdit);
  ranges_.back()->setup(node_,ranges_.size()-1);
  ranges_.back()->setMaxPosition(patchTester_.getPingVect().size()-1);
  ui->ranges_container->addWidget(ranges_.back());
  connect(ranges_.back(),SIGNAL(onPingChanged(int)),this,SLOT(on_range_edit(int)));
}

void PatchTesterGUI::on_range_edit(int index){
  //std::cout << index << std::endl;
  last_adjusted_index_=index;
  previewRange(index);
}

void PatchTesterGUI::on_bagfile_btn_clicked()
{
  QString fileName = QFileDialog::getOpenFileName(this,
      tr("Specify bag file"), "/home/", tr("Bag configs (*.bag)"));

  ui->bagfile_filename->setText(fileName);
  ui->bagfile_filename->setEnabled(false);
  _runThreads.push_back(std::thread(&PatchTesterGUI::load,this));

}

void PatchTesterGUI::on_urdf_btn_clicked()
{
  QString fileName = QFileDialog::getOpenFileName(this,
      tr("Specify URDF file"), "/home/", tr("URDF (*.urdf)"));
  ui->urdf_filename->setText(fileName);
  patchTester_.readRobotModel(ui->urdf_filename->text().toStdString());

  std::vector<urdf::LinkSharedPtr> links;
  patchTester_.getRobotModel().getLinks(links);
  for(size_t i = 1 ; i < links.size(); i++){
    ui->frame_selector->addItem(QString::fromUtf8(links[i]->name.c_str()));
    ui->save_trajectory_select->addItem(QString::fromUtf8(links[i]->name.c_str()));
  }
}

void PatchTesterGUI::on_project_btn_clicked()
{
  ui->project_btn->setEnabled( false );

  _runThreads.push_back(std::thread(&PatchTesterGUI::project,this));
}

void PatchTesterGUI::project(){
  patchTester_.projectPointsThreaded();
  std::cout << "done" << std::endl;
  cloudPub_.publish(patchTester_.getProjectedMsg());
  ui->project_btn->setEnabled( true );
}
void PatchTesterGUI::load(){
  this->setEnabled(false);
  patchTester_.openBag(ui->bagfile_filename->text().toStdString());
  ui->project_btn->setEnabled( true );
  for(rosbag::ConnectionInfo info:patchTester_.pointcloudConnections){
    ui->pointcloud_topics->addItem(QString::fromStdString(info.topic));
  }
  this->setEnabled(true);

}

void PatchTesterGUI::readPoints(){
  this->setEnabled(false);
  patchTester_.readBag(ui->pointcloud_topics->currentText().toStdString());

  for(std::string id : patchTester_.frame_ids){
      ui->map_frame->addItem(QString::fromStdString(id));
  }

  for(PingRangeEdit* range_edit: ranges_){
      range_edit->setMaxPosition(patchTester_.getPingVect().size()-1);
  }

  this->setEnabled(true);
}

void PatchTesterGUI::progressCallback(){
  QString errorTxt = QString::number(patchTester_.projectionFailures());
  ui->projection_errors->setText(errorTxt);
  ui->progressBar->setValue(patchTester_.getProgress());
  ros::spinOnce();
}

void PatchTesterGUI::on_save_btn_clicked()
{
  QString fileName = QFileDialog::getSaveFileName(this,
      tr("save as pcd"), "/home/", tr("PCD (*.pcd)"));
  patchTester_.savePCD(fileName.toStdString());
}

void PatchTesterGUI::on_frame_selector_currentIndexChanged(const QString &arg1)
{
    std::string linkName = arg1.toStdString();
    double roll,pitch,yaw,x,y,z;
    urdf::JointSharedPtr joint = patchTester_.getRobotModel().getLink(linkName)->parent_joint;
    joint->parent_to_joint_origin_transform.rotation.getRPY(roll,pitch,yaw);
    x = joint->parent_to_joint_origin_transform.position.x;
    y = joint->parent_to_joint_origin_transform.position.y;
    z = joint->parent_to_joint_origin_transform.position.z;

    ui->roll->setValue(roll*180/3.14159);
    ui->pitch->setValue(pitch*180/3.14159);
    ui->yaw->setValue(yaw*180/3.14159);

    ui->x_off->setValue(x);
    ui->y_off->setValue(y);
    ui->z_off->setValue(z);
}

void PatchTesterGUI::setJoint(){
    std::string linkName = ui->frame_selector->currentText().toStdString();
    urdf::JointSharedPtr joint = patchTester_.getRobotModel().getLink(linkName)->parent_joint;
    joint->parent_to_joint_origin_transform.rotation.setFromRPY(
                ui->roll->value()*3.14159/180,
                ui->pitch->value()*3.14159/180,
                ui->yaw->value()*3.14159/180
                );
    joint->parent_to_joint_origin_transform.position.x = ui->x_off->value();
    joint->parent_to_joint_origin_transform.position.y = ui->y_off->value();
    joint->parent_to_joint_origin_transform.position.z = ui->z_off->value();
    patchTester_.updateStaticTf();

}

void PatchTesterGUI::on_roll_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_pitch_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_yaw_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_x_off_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_y_off_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_z_off_valueChanged(double arg1)
{
    setJoint();
    std::vector<std::thread> threads;
    threads.push_back(std::thread(&PatchTesterGUI::updateCurrentPing,this));
    threads.push_back(std::thread(&PatchTesterGUI::updateComparePing,this));
    for (auto& th : threads) th.join();
}

void PatchTesterGUI::on_spinBox_valueChanged(int arg1)
{

}

void PatchTesterGUI::updateCurrentPing(){
    previewRange(last_adjusted_index_);
}

bool PatchTesterGUI::canProject(int index){
    if(!patchTester_.getRobotModel().getRoot()){
        return false;
    }
    ros::Time pingTime = patchTester_.getPing(ranges_[index]->getPingNo())->header.stamp;
    std::string base_link = patchTester_.getRobotModel().getRoot()->name;
    if(patchTester_.getBuffer().canTransform(patchTester_.getMapFrame(),base_link,pingTime)&&patchTester_.getRobotModel().getRoot()){
        return true;
    }else{
        ROS_WARN("can't transform from map frame '%s' to URDF base frame '%s'",patchTester_.getMapFrame().c_str(),base_link.c_str());
        return false;
    }
}

sensor_msgs::PointCloud2::Ptr PatchTesterGUI::getPreviewRangeMsg(int index){
    if(canProject(index)){
        return patchTester_.getProjectedMsg(ranges_[index]->getRangeMin(),
                                            ranges_[index]->getRangeMax());
    }else{
        sensor_msgs::PointCloud2::Ptr empty;
        empty.reset(new sensor_msgs::PointCloud2);
        return empty;
    }
}

void PatchTesterGUI::previewRange(int index){
    if(canProject(index)){
        int arg1 = ranges_[index]->getPingNo();
        ros::Time pingTime = patchTester_.getPing(arg1)->header.stamp;
        std::string base_link = patchTester_.getRobotModel().getRoot()->name;
        geometry_msgs::TransformStamped transform =
                patchTester_.getBuffer().lookupTransform(patchTester_.getMapFrame(),base_link,
                                                         pingTime);
        transform.header.stamp = ros::Time::now();
        tfBroadcaster_.sendTransform(transform);
        currentPingPub_.publish(getPreviewRangeMsg(index));
    }
}

void PatchTesterGUI::on_preview_ranges_btn_clicked(){
    updateComparePing();
}

void PatchTesterGUI::updateComparePing(){
    sensor_msgs::PointCloud2::Ptr out_cloud;
    out_cloud.reset(new sensor_msgs::PointCloud2);
    for (int i =0;i<int(ranges_.size());i++) {
        pcl::concatenatePointCloud(*out_cloud,*getPreviewRangeMsg(i),*out_cloud);
    }
    comparePingPub_.publish(out_cloud);
}

void PatchTesterGUI::on_save_trajectory_clicked()
{
    QString fileName = QFileDialog::getSaveFileName(this,
        tr("save as pcd"), "/home/untitled.pcd", tr("PCD (*.pcd)"));

    patchTester_.saveFrameTrack(ui->save_trajectory_select->currentText().toStdString(),fileName.toStdString());
}

void PatchTesterGUI::on_read_points_btn_clicked()
{
    _runThreads.push_back(std::thread(&PatchTesterGUI::readPoints,this));
}

void PatchTesterGUI::on_map_frame_currentTextChanged(const QString &arg1)
{
    patchTester_.setMapFrame(arg1.toStdString());
}

void PatchTesterGUI::on_add_range_btn_clicked()
{
    pushBackRange();
}


