#include "../include/sensor_stream/cudamat.h"
#include "../include/sensor_stream/cudamat_operations.h"
int main(){
    CudaMat<float> x1(4,3);
    x1(0,0)=1;x1(0,1)=2;x1(0,2)=3;
    x1(1,0)=4;x1(1,1)=5;x1(1,2)=6;
    x1(2,0)=7;x1(2,1)=8;x1(2,2)=9;
    x1(3,0)=7;x1(3,1)=8;x1(3,2)=9;
    x1.printMathematica();
    x1.host2dev();
//    ss::choleskyDecompose(x1);
//    x1.dev2host();
//    x1.printHost();


    CudaMat<float> out;

    out = ss::multDiagonal(x1,x1,CUBLAS_OP_N,CUBLAS_OP_T);

    out.dev2host();
    out.printMathematica();
//    CudaMat<float> x2(2,4);
//    x2(0,0)=1;x2(0,1)=4;x2(0,2)=3;;
//    x2(1,0)=2;x2(1,1)=5;x2(1,2)=6;

//    x2.printHost();
//    x2.host2dev();

//    x1.addRowsDev(2);
//    x1.addColsDev(3);
//    x1.dev2host();
//    x1.printHost();

//    x1.insertDev(x2,3,3);
//    x1.dev2host();
//    x1.printHost();


    //std::cout << info << std::endl;
//    CudaMat<float> x1(2,2);
//    CudaMat<float> x2(2,2);
//    CudaMat<float> x3(2,2);
//    CudaMat<float> result;

//    x1(0,0)=-1;
//    x1(1,0)=3.0/2.0;
//    x1(0,1)=1;
//    x1(1,1)=-1;
//    x1.printHost();
//    x1.printHostRawData();

//    x2(0,0)=5;
//    x2(1,0)=6;
//    x2(0,1)=7;
//    x2(1,1)=8;
//    x2.printHost();

//    x3(0,0)=9;
//    x3(1,0)=10;
//    x3(0,1)=11;
//    x3(1,1)=12;
//    x3.printHost();

//    x1.host2dev();
//    invert(x1);
//    x1.dev2host();
//    x1.printHost();

//    x1=x2.deepCopy();
//    x2(0,0)=99;
//    x1.printHost();
//    x2.printHost();


//    x1.host2dev();
//    x2.host2dev();
//    x3.host2dev();

//    result = x1*x2*x3;//mult(x1,x2);
//    result.dev2host();
//    result.printHost();

//    x1=x2.deepCopy();
//    x2(0,0)=99;

//    //add(x1,x2);
//    x1-=x2;
//    x1.dev2host();
//    x1.printHost();
    return 0;
}
