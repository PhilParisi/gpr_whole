#include "regression_params_widget.h"
#include "ui_regression_params_widget.h"

RegressionParamsWidget::RegressionParamsWidget(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::RegressionParamsWidget)
{
  ui->setupUi(this);
  setupSignals();
}

RegressionParamsWidget::~RegressionParamsWidget()
{
  delete ui;
}

void RegressionParamsWidget::setupSignals(){
  connect(ui->sensor_var, SIGNAL(valueChanged(double)), this, SLOT(updateParams()));
  connect(ui->nnz_thresh, SIGNAL(valueChanged(double)), this, SLOT(updateParams()));

}


void RegressionParamsWidget::updateParams(){
  if(params_){
    params_->nnzThresh = ui->nnz_thresh->value();
    params_->sensor_var = ui->sensor_var->value();
    for(auto hp_index: params_->kernel->getHPIndexMap()){
      std::string key = hp_index.first;
      if(hp_widget_map_.size()>0)
        params_->kernel->hyperparam(key)=hp_widget_map_[key]->getValue();
    }
    params_->kernel->hyperparam2dev(); // whenever the hyperparams are changed we need to send them to the device
  }
}

void RegressionParamsWidget::updateUI(){
  if(params_){
    setEnabled(true);
    ui->kernel_type->setText(QString::fromStdString(params_->kernel->getType()+" Hyperparameters:"));
    ui->nnz_thresh->setValue(params_->nnzThresh);
    ui->sensor_var->setValue(params_->sensor_var);

    for(auto hp_index: params_->kernel->getHPIndexMap()){
      std::string key = hp_index.first;
      if(hp_widget_map_[key]==nullptr){
        hp_widget_map_[key] = new KeyValueWidget;
        hp_widget_map_[key]->setKey(QString::fromStdString(key));
        ui->hyperparams->addWidget(hp_widget_map_[key]);
        hp_widget_map_[key]->setValue(params_->kernel->hyperparam(key));
        connect(hp_widget_map_[key], SIGNAL(valueChanged(double)), this, SLOT(updateParams()));
      }else {
        hp_widget_map_[key]->setValue(params_->kernel->hyperparam(key));
      }

    }

  }else{
    setEnabled(false);
  }
}
