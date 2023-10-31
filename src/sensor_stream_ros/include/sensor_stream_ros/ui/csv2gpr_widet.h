#ifndef CSV2GPR_WIDET_H
#define CSV2GPR_WIDET_H

#include <QWidget>
#include "csv2gpr_worker.h"
#include <QFileDialog>

namespace Ui {
class CSV2GPRWidet;
}

class CSV2GPRWidet : public QWidget
{
  Q_OBJECT

public:
  explicit CSV2GPRWidet(QWidget *parent = nullptr);
  ~CSV2GPRWidet();

public slots:
  void on_open_csv_btn_clicked();
  void on_predict_btn_clicked();
  void onMouseMove(QMouseEvent *event);
  void onPred1dFinished(gpr::Prediction pred);

  void on_generate_lml_clicked();
  void updateLMLRange(double min, double max);
  void updateLMLColorscale();
  void setMaxBlockSize(int size);

  //void on_lml_min_valueChanged(double arg1);

private:
  Ui::CSV2GPRWidet *ui;
  CSV2GPRWorker *worker_;
  QCPColorMap *color_map_;
};

#endif // CSV2GPR_WIDET_H
