#include "ui/patch_tester_gui.h"
#include <QApplication>
#include <QtCore/qglobal.h>


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "patchtester_node");
    QApplication a(argc, argv);
    PatchTesterGUI w;
    w.show();

    return a.exec();
}
