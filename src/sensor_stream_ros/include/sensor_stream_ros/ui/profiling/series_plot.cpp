#include "series_plot.h"
//#include "ui_series_plot.h"

// source: https://flatuicolors.com/palette/us
static const QColor colors[] = {
  QColor(9, 132, 227),
  QColor(214, 48, 49),
  QColor(0, 184, 148),
  QColor(108, 92, 231),
  QColor(253, 203, 110),
  QColor(0, 206, 201),
};

static const size_t num_colors = 6;

SeriesPlot::SeriesPlot(QWidget *parent) :
  QCustomPlot (parent)
{
  //ui->setupUi(this);
  setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes |
                                  QCP::iSelectLegend | QCP::iSelectPlottables);
  legend->setVisible(true);
  QFont legendFont = font();
  legendFont.setPointSize(10);
  legend->setFont(legendFont);
  legend->setSelectedFont(legendFont);
  legend->setSelectableParts(QCPLegend::spItems); // legend box shall not be selectable, only legend items

}

SeriesPlot::~SeriesPlot()
{
  delete ui;
}

void SeriesPlot::addGraph(std::string x_key, std::string y_key){
  GraphKey key(x_key,y_key);
  if (!hasGraph(x_key,y_key)){
    graph_key_map_.insert({key,GraphWithIterator(QCustomPlot::addGraph())});
    //graph_key_map_.left.count(key)=QCustomPlot::addGraph();
    //graph_keys_.push_back(key);
    graph(key)->setPen(QPen(colors[(graph_key_map_.size()-1)%num_colors],3));
    graph(key)->setName(QString::fromStdString(y_key));
    addErrorBars(key);
    xAxis->setLabel(QString::fromStdString(x_key));
  }
  update();
}

void SeriesPlot::addErrorBars(GraphKey key){
  if (series_ptr_->hasVariance(key.y_key)){
    QColor color = graph(key.getValue())->pen().color();
    color.setAlpha(20);
    if (!hasGraph(key.getLower())){
      graph_key_map_.insert({key.getLower(),GraphWithIterator(QCustomPlot::addGraph())});
      //graph_key_map_.left[key.getLower()]=QCustomPlot::addGraph();
      graph(key.getLower())->setPen(QPen(color));
      legend->removeItem(legend->itemCount()-1);
    }
    if (!hasGraph(key.getUpper())){
      graph_key_map_.insert({key.getUpper(),GraphWithIterator(QCustomPlot::addGraph())});
      //graph_key_map_.left.cou[key.getUpper()]=QCustomPlot::addGraph();
      graph(key.getUpper())->setPen(QPen(color));
      graph(key.getUpper())->setBrush(QBrush(color));
      graph(key.getUpper())->setChannelFillGraph(graph(key.getLower()));
      legend->removeItem(legend->itemCount()-1);
    }
  }
}

bool SeriesPlot::hasGraph(std::string x_key, std::string y_key){
  GraphKey key(x_key,y_key);
  return graph_key_map_.left.count(key)>0;
}

bool SeriesPlot::hasGraph(GraphKey key){
  return graph_key_map_.left.count(key)>0;
}

QCPGraph * SeriesPlot::graph(GraphKey key){
  return graph_key_map_.left.at(key).graph;
}

size_t & SeriesPlot::lastIndex(GraphKey key){
  return *graph_key_map_.left.at(key).last_index;
}

GraphKey SeriesPlot::key(QCPGraph* graph){
  return graph_key_map_.right.at(graph);
}

void SeriesPlot::update(){
  for (auto kv : graph_key_map_) {
    GraphKey key = kv.left;
    //graph(key)->data().clear();
    if(key.type==graph_type::value){
      for (size_t i = lastIndex(key); i < series_ptr_->size() ; i++) {
          graph(key)->addData(series_ptr_->getValue(key.x_key,i),series_ptr_->getValue(key.y_key,i));
          lastIndex(key) = series_ptr_->size();
      }
      if(series_ptr_->hasVariance(key.y_key)){
        for (size_t i = 0; i < series_ptr_->size() ; i++) {
          //graph(key.getUpper())->data().clear();
          //graph(key.getLower())->data().clear();
          double value = series_ptr_->getValue(key.y_key,i);
          double stdev = sqrt(series_ptr_->getVariance(key.y_key,i));
          double upper = value+stdev;
          double lower = value-stdev;
          graph(key.getUpper())->addData(series_ptr_->getValue(key.x_key,i), upper);
          graph(key.getLower())->addData(series_ptr_->getValue(key.x_key,i),lower);
          lastIndex(key) = series_ptr_->size();
        }
      }
      rescaleAxes();
      replot();
    }
  }
}

void SeriesPlot::removeGraph(GraphKey key){
  if(hasGraph(key.getUpper())){
    QCustomPlot::removeGraph(graph(key.getUpper()));
    graph_key_map_.left.erase(key.getUpper());
  }
  if(hasGraph(key.getValue())){
    QCustomPlot::removeGraph(graph(key.getValue()));
    graph_key_map_.left.erase(key.getValue());
  }
  if(hasGraph(key.getLower())){
    QCustomPlot::removeGraph(graph(key.getLower()));
    graph_key_map_.left.erase(key.getLower());
  }
  replot();
  return;
}

void SeriesPlot::setColor(GraphKey key, QColor color){
  auto bg_color = color;
  bg_color.setAlphaF(color.alphaF()*0.2);
  if(hasGraph(key.getUpper())){
    graph(key.getUpper())->setPen(bg_color);
    graph(key.getUpper())->setBrush(bg_color);
  }
  if(hasGraph(key.getValue())){
    graph(key.getValue())->setPen(QPen(color,3));
  }
  if(hasGraph(key.getLower())){
    graph(key.getLower())->setPen(QPen(bg_color));
  }
  replot();
}
