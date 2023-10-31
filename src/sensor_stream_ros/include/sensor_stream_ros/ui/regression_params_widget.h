#ifndef REGRESSION_WIDGET_H
#define REGRESSION_WIDGET_H

#include <QWidget>
#include <sensor_stream/include/sensor_stream/blockgpr.h>
#include "key_value_widget.h"
#include <map>

namespace Ui {
class RegressionParamsWidget;
}

class RegressionParamsWidget : public QWidget
{
  Q_OBJECT

public:
  explicit RegressionParamsWidget(QWidget *parent = nullptr);
  ~RegressionParamsWidget();
  void setRegressionParams(gpr::RegressionParams::ptr params){params_=params; updateUI();}
  void setupSignals();


public slots:
  void updateParams();
  void updateUI();


private:
  Ui::RegressionParamsWidget *ui;
  gpr::RegressionParams::ptr params_;
  std::map<std::string,KeyValueWidget*> hp_widget_map_;
};

#endif // REGRESSION_WIDGET_H
