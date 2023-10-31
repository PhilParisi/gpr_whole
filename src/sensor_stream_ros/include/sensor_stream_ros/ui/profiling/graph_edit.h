#ifndef GRAPH_EDIT_H
#define GRAPH_EDIT_H

#include <QWidget>
#include "series_plot.h"

namespace Ui {
class GraphEdit;
}

class KeyListItem : public QListWidgetItem{
public:
  GraphKey key;
  KeyListItem(GraphKey k):key(k){return;}

};

class GraphEdit : public QWidget
{
  Q_OBJECT

public:
  explicit GraphEdit(QWidget *parent = nullptr);
  ~GraphEdit();
  void setPlot(SeriesPlot * plot);
  void setSelected(QCPGraph * graph);
  GraphKey  getSelectedKey();
  QCPGraph *  getSelectedGraph();
public slots:
  void update();


private slots:
  void on_remove_clicked();

  void on_color_btn_clicked();

  void on_line_style_currentIndexChanged(int index);

private:
  void updateKeys();
  Ui::GraphEdit *ui;
  SeriesPlot * plot_;
  std::vector<GraphKey> keys_;
};

#endif // GRAPH_EDIT_H
