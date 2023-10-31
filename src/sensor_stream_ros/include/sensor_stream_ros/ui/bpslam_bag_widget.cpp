#include "bpslam_bag_widget.h"
#include "ui_bpslam_bag_widget.h"

BPSlamBagWidget::BPSlamBagWidget(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::BPSlamBagWidget)
{
  ui->setupUi(this);
  worker_thread_ = new QThread;
  setupSignals();
  checkVaildFile();


}

BPSlamBagWidget::~BPSlamBagWidget()
{
  delete ui;
//  if(worker_!=nullptr){
//    delete  worker_;
//    worker_ = nullptr;
//  }
}

void BPSlamBagWidget::setupSignals(){
  connect(ui->bag_file, SIGNAL(textChanged(QString)), this, SLOT(checkVaildFile()));
  connect(ui->urdf_file, SIGNAL(textChanged(QString)), this, SLOT(checkVaildFile()));
}

void BPSlamBagWidget::setBagProcessor(ss::bpslam::BagProcessor::Ptr bag_processor_ptr){
  bag_processor_ptr_=bag_processor_ptr;
  ui->bpslam_widget->setBPSlam(bag_processor_ptr_->pf_);
}

void BPSlamBagWidget::on_urdf_button_clicked()
{
  QString fileName = QFileDialog::getOpenFileName(this, tr("Specify A URDF File"),
                                                  "/home",
                                                  tr("urdf File (*.urdf)"));
  ui->urdf_file->setText(fileName);
}

void BPSlamBagWidget::on_bag_button_clicked()
{
  QString fileName = QFileDialog::getOpenFileName(this, tr("Specify A Bag File"),
                                                  "/home",
                                                  tr("Bag File (*.bag)"));
  ui->bag_file->setText(fileName);
}

void BPSlamBagWidget::checkVaildFile(){
  if(   boost::filesystem::exists(ui->bag_file->text().toStdString())
     && boost::filesystem::exists(ui->urdf_file->text().toStdString())
        ){
    ui->run_btn->setEnabled(true);
    ui->run_btn->setText("Run");
  }else {
    ui->run_btn->setEnabled(false);
    ui->run_btn->setText("Specify valid input files");
  }
}

void BPSlamBagWidget::on_run_btn_clicked()
{
  worker_thread_->start();
}

void BPSlamBagWidget::setBagFile(QString bag_file){
  ui->bag_file->setText(bag_file);
}

void BPSlamBagWidget::setUrdfFile(QString urdf_file){
  ui->urdf_file->setText(urdf_file);
}

void BPSlamBagWidget::on_start_btn_clicked()
{
  try {
    bag_processor_ptr_->openBag(ui->bag_file->text().toStdString());
    bag_processor_ptr_->loadURDF(ui->urdf_file->text().toStdString());
    bag_processor_ptr_->readBag();
    worker_ = new BPSLAMBagWorker(bag_processor_ptr_);
    worker_->moveToThread(worker_thread_);

    connect(ui->run_btn, SIGNAL(clicked()), worker_, SLOT(start()));
    connect(ui->pause_btn, SIGNAL(clicked()), worker_, SLOT(pause()));
    connect(ui->step_btn, SIGNAL(clicked()), worker_, SLOT(stepOnce()));
    connect(ui->cull_btn, SIGNAL(clicked()), worker_, SLOT(cullParticles()));
    connect(ui->gpr_btn, SIGNAL(clicked()), worker_, SLOT(computeLeafsGPR()));
    connect(worker_, SIGNAL(ready2spawn()), ui->bpslam_widget, SLOT(updateUI()));
    connect( ui->bpslam_widget, SIGNAL(uiUpdateComplete()), worker_, SLOT(onGuiUpdate()));
  } catch (...) {
    QMessageBox msgBox;
    msgBox.setText(
          QString::fromStdString(
            boost::current_exception_diagnostic_information()
          )
    );
    msgBox.exec();
  }
}
