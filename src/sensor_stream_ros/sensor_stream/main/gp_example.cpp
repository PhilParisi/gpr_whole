#include <QtWidgets/QApplication>
#include <QtWidgets/QMainWindow>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include "../include/sensor_stream/gpr.h"

QT_CHARTS_USE_NAMESPACE

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    Gpr testGP;
    size_t div=1000;
    size_t samples=1000;
    float upper,lower;
    upper=10;
    lower=0;
    CudaMat<float> hp(1,1);
    hp(0,0)=.8;
    hp.host2dev();



    testGP.genPredVect(lower,upper,div);

    CudaMat<float> x(samples,1);
    for(size_t i = 0 ; i<x.size() ; i++)
        x(i,0) = std::rand()/((RAND_MAX)/10.0);


    CudaMat<float> y(samples,1);
    for(size_t i = 0; i<x.size(); i++){
        y(i,0) = sin( x(i,0));
    }

    QScatterSeries  *training = new QScatterSeries();
    for(size_t i = 1 ; i<y.size() ; i++){
        training->append(x(i,0), y(i,0));
    }
    x.host2dev();
    y.host2dev();


    testGP.setTrainingData(x,y);
    testGP.setKernel(&covMatKernel);
    testGP.setHyperParam(hp);
    testGP.solve();
    testGP._mu.dev2host();
    testGP._var.dev2host();


    QLineSeries *series = new QLineSeries();
    QPen pen = series->pen();
    pen.setWidth(4);
    pen.setBrush(QBrush("red")); // or just pen.setColor("red");
    series->setPen(pen);

    for(size_t i = 0 ; i<testGP._mu.size() ; i++){
        series->append(i*((upper-lower)/float(div-1)), testGP._mu(i,0));
    }

    QLineSeries *bound1 = new QLineSeries();

    for(size_t i = 1 ; i<testGP._mu.size() ; i++){
        bound1->append(i*((upper-lower)/float(div-1)), testGP._var(i,i)+testGP._mu(i,0));
    }

    QLineSeries *bound2 = new QLineSeries();
    for(size_t i = 1 ; i<testGP._mu.size() ; i++){
        bound2->append(i*((upper-lower)/float(div-1)), -testGP._var(i,i)+testGP._mu(i,0));
    }



    QChart *chart = new QChart();
    chart->legend()->hide();

    chart->addSeries(bound1);
    chart->addSeries(bound2);
    chart->addSeries(series);
    chart->addSeries(training);
    chart->createDefaultAxes();
    chart->setTitle("GP Example");

    QChartView *chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    QMainWindow window;
    window.setCentralWidget(chartView);
    window.resize(1500, 1000);
    window.show();

    return a.exec();
}
