#include "csv2gpr_widet.h"
#include "ui_csv2gpr_widet.h"

CSV2GPRWidet::CSV2GPRWidet(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::CSV2GPRWidet)
{
  ui->setupUi(this);
  worker_ = new CSV2GPRWorker;

  ui->params_widget->setRegressionParams(worker_->getParams().regression);

  connect(worker_, SIGNAL(predict1dFinished(gpr::Prediction)), this, SLOT(onPred1dFinished(gpr::Prediction)));
  connect(worker_, SIGNAL(lmlComplete(double,double)), this, SLOT(updateLMLRange(double,double)));
  connect(ui->lml_plot, SIGNAL(mouseMove(QMouseEvent*)), this, SLOT(onMouseMove(QMouseEvent*)));
  connect(ui->block_size, SIGNAL(valueChanged(int)),worker_,SLOT(setBlockSize(int)));
  connect(worker_, SIGNAL(dataLoaded(int)),this, SLOT(setMaxBlockSize(int)));

}

CSV2GPRWidet::~CSV2GPRWidet()
{
  delete ui;
}

void CSV2GPRWidet::on_open_csv_btn_clicked()
{
  QString fileName = QFileDialog::getOpenFileName(this, tr("Specify A CSV File"),
                                                  "/home",
                                                  tr("Bag File (*.csv)"));
  worker_->readFile(fileName);
}

void CSV2GPRWidet::setMaxBlockSize(int size){
  ui->block_size->setMaximum(size);
}

void CSV2GPRWidet::on_predict_btn_clicked()
{
    worker_->predict1d(ui->x_min->value(),ui->x_max->value(),ui->divisions->value(),ui->y_val->value());
}

void CSV2GPRWidet::onPred1dFinished(gpr::Prediction pred){

  ui->pred_plot->clearGraphs();
  ui->pred_plot->addGraph();
  ui->pred_plot->graph(0)->setPen(QPen(Qt::red)); // line color blue for first graph
  ui->pred_plot->addGraph();
  ui->pred_plot->graph(1)->setPen(QPen(QColor(255,50,30,20)));
  ui->pred_plot->graph(1)->setBrush(QBrush(QColor(255,50,30,20)));
  ui->pred_plot->addGraph();
  ui->pred_plot->legend->removeItem(ui->pred_plot->legend->itemCount()-1); // don't show two confidence band graphs in legend
  ui->pred_plot->graph(2)->setPen(QPen(QColor(255,50,30,20)));
  ui->pred_plot->graph(1)->setChannelFillGraph(ui->pred_plot->graph(2));

  ui->pred_plot->addGraph();
  ui->pred_plot->graph(3)->setPen(QPen(Qt::blue));
  ui->pred_plot->graph(3)->setLineStyle(QCPGraph::lsNone);
  ui->pred_plot->graph(3)->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, 4));
  // add error bars:
  QCPErrorBars *errorBars = new QCPErrorBars(ui->pred_plot->xAxis, ui->pred_plot->yAxis);
  errorBars->removeFromLegend();
  errorBars->setAntialiased(false);
  errorBars->setDataPlottable(ui->pred_plot->graph(3));
  errorBars->setPen(QPen(QColor(180,180,180)));
  ui->pred_plot->graph(3)->setName("Measurement");

  QVector<double> x(pred.mu.size()), y0(pred.mu.size());
  for (int i=0; i<pred.mu.size(); ++i)
  {
    x[i]  = pred.points.x(i);
    y0[i] = pred.mu(i);

    double confidence = 1.648*sqrt(pred.sigma(i));

    ui->pred_plot->graph(1)->addData(pred.points.x(i),pred.mu(i)+confidence);
    ui->pred_plot->graph(2)->addData(pred.points.x(i),pred.mu(i)-confidence);
  }

  size_t training_size = worker_->getGpr()->getXTrain().rows()*worker_->getGpr()->getXTrain().getBlockParam().rows;
  QVector<double> x1(training_size), y1(training_size), y1err(training_size);
  for (int i=0; i<training_size; ++i)
  {
    x1[i] = worker_->getGpr()->getXTrain().getVal(i,0);
    y1[i] = worker_->getGpr()->getYTrain().getVal(i,0);;
    y1err[i] = sqrt(worker_->getGpr()->params.regression->sensor_var);
  }

  ui->pred_plot->xAxis2->setVisible(true);
  ui->pred_plot->xAxis2->setTickLabels(false);
  ui->pred_plot->yAxis2->setVisible(true);
  ui->pred_plot->yAxis2->setTickLabels(false);

  ui->pred_plot->graph(0)->setData(x, y0);

  ui->pred_plot->graph(3)->setData(x1, y1);
  errorBars->setData(y1err);

  ui->pred_plot->rescaleAxes();

  ui->pred_plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
  ui->pred_plot->replot();

}

void CSV2GPRWidet::onMouseMove(QMouseEvent *event){
  QPoint p = event->pos();
  double X = ui->lml_plot->xAxis->pixelToCoord(p.x());
  double Y = ui->lml_plot->yAxis->pixelToCoord(p.y());
  QString text;
  text.sprintf("Cursor Location:  x=%04.1f, y=%04.1f", X,Y);
  ui->pos_lbl->setText(text);

}

void CSV2GPRWidet::on_generate_lml_clicked()
{
  ui->lml_plot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
  ui->lml_plot->axisRect()->setupFullAxesBox(true);



  color_map_ = new QCPColorMap(ui->lml_plot->xAxis, ui->lml_plot->yAxis);
//  worker_->genLML(ui->lml_plot,color_map_,
//                  -2,
//                  2,
//                  -2,
//                  4,
//                  100,
//                  100);

  worker_->genLML(ui->lml_plot,color_map_,
                  ui->hp_x_min->value(),
                  ui->hp_x_max->value(),
                  ui->hp_y_min->value(),
                  ui->hp_y_max->value(),
                  ui->hp_x_div->value(),
                  ui->hp_y_div->value());
}

void CSV2GPRWidet::updateLMLRange(double min,double max){
  ui->lml_min->setRange(min,max);
  ui->lml_min->setValue(min);
  ui->lml_max->setRange(min,max);
  ui->lml_max->setValue(max);
  connect(ui->lml_min, SIGNAL(valueChanged(double)), this, SLOT(updateLMLColorscale()));
  connect(ui->lml_max, SIGNAL(valueChanged(double)), this, SLOT(updateLMLColorscale()));
}

void CSV2GPRWidet::updateLMLColorscale(){
  if(color_map_){
    QCPRange range(ui->lml_min->value(),ui->lml_max->value());
    color_map_->setDataRange(range);
    ui->lml_plot->rescaleAxes();
    ui->lml_plot->replot();
  }
}

