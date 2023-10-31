#include "graph_edit.h"
#include "ui_graph_edit.h"

GraphEdit::GraphEdit(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::GraphEdit)
{
  ui->setupUi(this);
  QStringList lineNames;
  lineNames << "None" << "Line" << "StepLeft" << "StepRight" << "StepCenter" << "Impulse";
  ui->line_style->addItems(lineNames);
}

GraphEdit::~GraphEdit()
{
  delete ui;
}

void GraphEdit::setPlot(SeriesPlot *plot){
  plot_=plot;
  update();
}

void GraphEdit::setSelected(QCPGraph *graph){
  GraphKey key = plot_->getKeyMap().right.at(graph);
  int index = ui->key->findText(QString::fromStdString(key.toString()));
  ui->key->setCurrentIndex(index);
}

GraphKey GraphEdit::getSelectedKey(){
  return keys_[ui->key->currentIndex()];
}

QCPGraph *  GraphEdit::getSelectedGraph(){
  return plot_->graph(getSelectedKey());
}

void GraphEdit::update(){
  updateKeys();
  if(keys_.size()<=0)
    return;

  ui->line_style->setCurrentIndex(getSelectedGraph()->lineStyle());
}

void GraphEdit::updateKeys(){
  ui->key->clear();
  keys_.clear();
  for (auto kv : plot_->getKeyMap()){
    GraphKey key = kv.left;
    if(key.type==graph_type::value){
      ui->key->addItem(QString::fromStdString(key.toString()));
      keys_.push_back(key);
    }
  }
}

void GraphEdit::on_remove_clicked()
{
  if(keys_.size()<=0)
    return;
  GraphKey key = keys_[ui->key->currentIndex()];
  plot_->removeGraph(key);
  update();
}

void GraphEdit::on_color_btn_clicked()
{
  if(keys_.size()<=0)
    return;
  GraphKey key = keys_[ui->key->currentIndex()];
  QColor color = QColorDialog::getColor(plot_->graph(key)->pen().color(), this);
  plot_->setColor(key,color);
}

void GraphEdit::on_line_style_currentIndexChanged(int index)
{
  if(keys_.size()<=0)
    return;
  getSelectedGraph()->setLineStyle((QCPGraph::LineStyle)index);
  plot_->replot();
}
