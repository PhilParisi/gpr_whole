#ifndef KEY_VALUE_WIDGET_H
#define KEY_VALUE_WIDGET_H

#include <QWidget>
#include <QString>

namespace Ui {
class KeyValueWidget;
}

class KeyValueWidget : public QWidget
{
  Q_OBJECT

public:
  explicit KeyValueWidget(QWidget *parent = nullptr);
  ~KeyValueWidget();
  double getValue();

signals:
  void valueChanged(double value);
public slots:
  void setKey(QString key);
  void setValue(double value);

private slots:
  void on_value_valueChanged(double arg1);

private:
  Ui::KeyValueWidget *ui;
};

#endif // KEY_VALUE_WIDGET_H
