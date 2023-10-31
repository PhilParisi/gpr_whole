#ifndef PING_RANGE_EDIT_H
#define PING_RANGE_EDIT_H

#include <QWidget>
#include <iostream>
#include <ros/ros.h>
namespace Ui {
class PingRangeEdit;
}

class PingRangeEdit : public QWidget
{
    Q_OBJECT

public:
    explicit PingRangeEdit(QWidget *parent = nullptr);
    ~PingRangeEdit();
    void setupConnections();
    int getRangeMin();
    int getRangeMax();
    int getPingNo(){return getRangeMax();}

    void setup(ros::NodeHandlePtr nh, int index){index_=index; node_=nh;}
    void setMaxPosition(int pos);

    signals:
    void onPingChanged(int index);

private slots:
    void maxPingChanged(int max);
    void minPingChanged(int min);

    void on_range_width_valueChanged(int arg1);

    void on_pingNum_valueChanged(int arg1);

private:
    Ui::PingRangeEdit *ui;
    int index_;
    ros::NodeHandlePtr node_;
};

#endif // PING_RANGE_EDIT_H
