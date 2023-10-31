#include "key_value_widget.h"
#include "ui_key_value_widget.h"

KeyValueWidget::KeyValueWidget(QWidget *parent) :
  QWidget(parent),
  ui(new Ui::KeyValueWidget)
{
  ui->setupUi(this);
}

KeyValueWidget::~KeyValueWidget()
{
  delete ui;
}

double KeyValueWidget::getValue(){
  return ui->value->value();
}

void KeyValueWidget::setKey(QString key){
  ui->key->setText(key);
}

void KeyValueWidget::setValue(double value){
  ui->value->setValue(value);
}

void KeyValueWidget::on_value_valueChanged(double arg1)
{
  emit valueChanged(arg1);
}
