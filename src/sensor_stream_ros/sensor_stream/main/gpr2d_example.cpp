#include "../include/sensor_stream/gpr.h"
#include <QtDataVisualization>

using namespace QtDataVisualization;

int main(int argc, char *argv[])
{

    Gpr testGP;
    Gpr GroundGP;
    size_t div=1000;
    size_t samples=200;
    float upper,lower;
    upper=10;
    lower=0;
    CudaMat<float> hp(1,1);
    hp(0,0)=3;  // set the hyperparameters
    hp.host2dev();
    testGP.setHyperParam(hp);
    testGP.genPredVect2d(lower,upper,div);
    testGP.setKernel(gpKernel2d);

    GroundGP.setHyperParam(hp);
    GroundGP.genPredVect2d(lower,upper,div);
    GroundGP.setKernel(gpKernel2d);

    CudaMat<float> x(samples*2,2); // samples rows, 2 colums to represent 2d point
    CudaMat<float> xall(samples*3,2);
    for(size_t i = 0 ; i<samples*2 ; i++){
        float rand1 = std::rand()/((RAND_MAX)/10.0);
        float rand2 = std::rand()/((RAND_MAX)/10.0);
        x(i,0) = rand1; // x point
        xall(i,0) =rand1;
        x(i,1) = rand2; // y point
        xall(i,1) =rand2;
    }

    x.host2dev();
    x.printHost();

    CudaMat<float> x2(samples,2); // samples rows, 2 colums to represent 2d point
    for(size_t i = 0 ; i<samples ; i++){
        float rand1 = std::rand()/((RAND_MAX)/10.0);
        float rand2 = std::rand()/((RAND_MAX)/10.0);
        x2(i,0) = rand1; // x point
        xall(i+samples*2,0) =rand1;
        x2(i,1) = rand2; // y point
        xall(i+samples*2,1) =rand2;
    }

    x2.host2dev();
    x2.printHost();
    xall.host2dev();
    xall.printHost();

    CudaMat<float> y(samples,1);
    for(size_t i = 0; i<y.size(); i++){
        y(i,0) = sin( x(i,0)+x(i,1));
    }
    y.host2dev();


//    testGP.addTrainingData(x,y);
//    testGP.addTrainingData(x2,y);
//    printf("ground \n");
//    GroundGP.addTrainingData(xall,y);


//    for(size_t j=0 ; j<30 ; j++){
//        CudaMat<float> x3(samples,2); // samples rows, 2 colums to represent 2d point
//        for(size_t i = 0 ; i<samples ; i++){
//            float rand1 = (std::rand()/((RAND_MAX)/10.0))+10*i;
//            float rand2 = (std::rand()/((RAND_MAX)/10.0))+10*i;
//            x3(i,0) = rand1; // x point
//            x3(i,1) = rand2; // y point
////            printf("%f,%f \n",x3(i,0),x3(i,0));

//        }

//        x3.host2dev();
//        std::cout << j << std::endl;
//        testGP.addTrainingData(x3,y);
//    }






//    CudaMat<float> x3(samples,2); // samples rows, 2 colums to represent 2d point
//    for(size_t i = 0 ; i<samples ; i++){
//        float rand1 = std::rand()/((RAND_MAX)/10.0);
//        float rand2 = std::rand()/((RAND_MAX)/10.0);
//        x3(i,0) = rand1; // x point
//        x3(i,1) = rand2; // y point

//    }

//    x3.host2dev();
//    testGP.addTrainingData(x3,y);


//    CudaMat<float> s11 = testGP.genCovariance(x,x,&gpKernel2d,hp);


//    ss::choleskyDecompose(s11);
//    s11.dev2host();
//    s11.printHost();

//    CudaMat<float> s12 = testGP.genCovariance(x,x2,&gpKernel2d,hp);
//    ss::choleskySolve(s11,s12,CUBLAS_FILL_MODE_LOWER);

//    CudaMat<float> s12_squared = mult(s12,s12,CUBLAS_OP_T,CUBLAS_OP_N);

//    CudaMat<float> s22 = testGP.genCovariance(x2,x2,&gpKernel2d,hp);
//    s22-=s12_squared;
//    ss::choleskyDecompose(s22);
//    s22.dev2host();
//    s22.printHost();

//    CudaMat<float> groundTruth= testGP.genCovariance(xall,xall,&gpKernel2d,hp);
//    ss::choleskyDecompose(groundTruth);
//    groundTruth.dev2host();
//    groundTruth.printHost();






//    testGP.setTrainingData(x,y);
//    testGP.setKernel(&gpKernel2d);
//    testGP.setHyperParam(hp);
//    testGP.solve();
//    testGP._mu.dev2host();
//    testGP._var.dev2host();
//    testGP._pred.dev2host();

//    QGuiApplication app(argc, argv);


//    Q3DScatter scatter;
//    scatter.setFlags(scatter.flags() ^ Qt::FramelessWindowHint);
//    QScatter3DSeries *series = new QScatter3DSeries;
//    QScatterDataArray data;
//    for(size_t i=0; i<div ; i++){
//        //QSurfaceDataRow *dataRow = new QSurfaceDataRow;
//        for(size_t j=0; j<div; j++){
//            size_t index = i+(j*div);
//            data << QVector3D(testGP._pred(index,0), testGP._pred(index,1), testGP._mu(index,0));
//        }
//        //*data << dataRow;
//    }

//    QScatter3DSeries *series2 = new QScatter3DSeries;
//    QScatterDataArray data2;
//    for(size_t i=0; i<samples ; i++){
//        data2 << QVector3D(x(i,0),x(i,1),y(i,0));
//    }

//    series->dataProxy()->addItems(data);
//    scatter.addSeries(series);
//    series2->dataProxy()->addItems(data2);
//    scatter.addSeries(series2);
//    scatter.show();
//    scatter.seriesList().at(1)->setBaseColor(Qt::blue);
//    scatter.seriesList().at(1)->setItemSize(.1);
//    QLinearGradient linearGrad(QPointF(0, -1), QPointF(0, 1));
//    linearGrad.setColorAt(0, Qt::white);
//    linearGrad.setColorAt(1, Qt::red);
//    scatter.seriesList().at(0)->setBaseGradient(linearGrad);
//    scatter.seriesList().at(0)->setColorStyle(Q3DTheme::ColorStyle::ColorStyleObjectGradient);
//    scatter.seriesList().at(0)->setItemSize(.1);
////    scatter.seriesList().at(0)->setMesh(QAbstract3DSeries::MeshPoint);

//    return app.exec();

    //testGP._mu.printHost();
}
