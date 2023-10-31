#include <sensor_stream_ros/ui/bpslam_widget.h>
#include <sensor_stream_ros/ui/bpslam_bag_widget.h>
#include <QApplication>
#include <QtCore/qglobal.h>


int main(int argc, char *argv[])
{
    ros::init(argc, argv, "bpslam_bag_gui");
    QApplication a(argc, argv);
    ss::bpslam::BagProcessor::Ptr processor_ptr(new ss::bpslam::BagProcessor);
    BPSlamBagWidget w;
    w.setBagProcessor(processor_ptr);
    w.setBagFile(argv[1]);
    w.setUrdfFile(argv[2]);
    w.show();

    return a.exec();
}
