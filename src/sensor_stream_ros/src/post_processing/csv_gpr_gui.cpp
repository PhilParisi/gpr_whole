
#include <sensor_stream_ros/ui/csv2gpr_widet.h>
#include <QApplication>
#include <QtCore/qglobal.h>


int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    CSV2GPRWidet w;
    w.show();

    return a.exec();
}
