#include "series_plot_editor.h"
#include "ui_series_plot_editor.h"




SeriesPlotEditor::SeriesPlotEditor(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::SeriesPlotEditor)
{
  ui->setupUi(this);
  connect(ui->plot, SIGNAL(selectionChangedByUser()), this, SLOT(selectionChanged()));
  ui->graph_editor->setPlot(ui->plot);
}

SeriesPlotEditor::~SeriesPlotEditor()
{
  delete ui;
}

void SeriesPlotEditor::setSeries(ss::profiling::Series::Ptr series){
  ui->plot->setSeries(series);
  series_= series;
}

void SeriesPlotEditor::on_add_plot_clicked()
{
  ui->plot->addGraph(ui->x_selection->currentText().toStdString(),ui->y_selection->currentText().toStdString());
  ui->graph_editor->update();
}

void SeriesPlotEditor::update(){
  for(auto key : series_->getKeys()){
    if(ui->x_selection->findText(QString::fromStdString(key)) < 0){
      ui->x_selection->addItem(QString::fromStdString(key));
      ui->y_selection->addItem(QString::fromStdString(key));
    }
  }
  ui->plot->update();
}


void SeriesPlotEditor::selectionChanged(){
  if(!ui->plot->selectedGraphs().isEmpty()){
    auto selected = ui->plot->selectedGraphs().first();
    ui->graph_editor->setSelected(selected);
  }
  return;
}
