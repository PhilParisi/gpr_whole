#include "ui/ping_range_edit.h"
#include "ui_ping_range_edit.h"

PingRangeEdit::PingRangeEdit(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PingRangeEdit)
{
    ui->setupUi(this);
    setupConnections();
    ui->range_width->setValue(100);
    ui->pingNum->setValue(100);
}

PingRangeEdit::~PingRangeEdit()
{
    delete ui;
}

void PingRangeEdit::setupConnections(){
    connect(ui->range_slider,SIGNAL(maximumValueChanged(int)),this,SLOT(maxPingChanged(int)));
    connect(ui->range_slider,SIGNAL(minimumValueChanged(int)),this,SLOT(minPingChanged(int)));
}

void PingRangeEdit::maxPingChanged(int max){
    ui->range_slider->setMinimumValue(max - ui->range_width->value());
    ui->pingNum->setValue(max);
    emit onPingChanged(index_);
}
void PingRangeEdit::minPingChanged(int min){
    ui->range_width->setValue(ui->range_slider->maximumValue()-min);
    emit onPingChanged(index_);
}

void PingRangeEdit::on_range_width_valueChanged(int arg1){
    ui->range_slider->setMinimumValue(ui->range_slider->maximumValue() - arg1);
}

void PingRangeEdit::on_pingNum_valueChanged(int arg1){
    ui->range_slider->setMaximumValue(arg1);
}

int PingRangeEdit::getRangeMin(){
    return ui->range_slider->minimumValue();
}

int PingRangeEdit::getRangeMax(){
    if(ui->range_slider->maximumValue()<0)
        return 0;
    else
        return ui->range_slider->maximumValue();
}

void PingRangeEdit::setMaxPosition(int pos){
    ui->range_slider->setMaximum(pos);
    ui->pingNum->setMaximum(pos);
    ui->range_width->setMaximum(pos);

    ui->range_slider->setMinimumValue(0);
    ui->pingNum->setMinimum(0);
    ui->range_width->setMinimum(1);
}
