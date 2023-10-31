#include "series_plot_array.h"
#include "ui_series_plot_array.h"

SeriesPlotArray::SeriesPlotArray(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::SeriesPlotArray)
{
  ui->setupUi(this);
}

SeriesPlotArray::~SeriesPlotArray()
{
  delete ui;
}

void SeriesPlotArray::addPlot(size_t index){
  SeriesPlotEditor * plot = new SeriesPlotEditor(this);
  plot->setSeries(report_->getSeriesVect()[index]);
  plots_.push_back(plot);
  ui->plot_layout->addWidget(plot);
  update();
}

void SeriesPlotArray::setReport(ss::profiling::Report::Ptr report){

  //series_.push_back(series);
  //ss::profiling::Report::addSeries(series);
  report_ = report;
  for(auto series: report_->getSeriesVect()){
    ui->series->addItem(QString::fromStdString( series->getName() ));
  }
  update();
}

void SeriesPlotArray::update(){
  for (auto plot : plots_) {
    plot->update();
  }
}

void SeriesPlotArray::on_add_plot_clicked()
{
  addPlot(ui->series->currentIndex());
}
