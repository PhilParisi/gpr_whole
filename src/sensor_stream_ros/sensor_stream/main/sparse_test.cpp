#include "../include/sensor_stream/cudasparsemat.h"
#include "../include/sensor_stream/gpr.h"
#include "../include/sensor_stream/sparsemat_operations.h"
#include "../include/sensor_stream/cudablockmat.h"
#include "../include/sensor_stream/abstractcsrmat.h"
#include "../include/sensor_stream/cudamat_operations.h"


int main(int argc, char *argv[])
{
    /*
    printf("declare blockMat \n");
    CudaBlockMat<float> blockMat;
    CudaBlockMat<float> B;
    printf("declare x1 \n");
    CudaMat<float> A0(2,2);
    A0(0,0)=1;A0(0,1)=0;
    A0(1,0)=2;A0(1,1)=1;

    CudaMat<float> A1(2,2);
    A1(0,0)=3;A1(0,1)=5;
    A1(1,0)=4;A1(1,1)=6;


    CudaMat<float> A2(2,2);
    A2(0,0)=1;A2(0,1)=0;
    A2(1,0)=7;A2(1,1)=1;

    CudaMat<float> B0(2,2);
    B0(0,0)=5;B0(0,1)=1;
    B0(1,0)=6;B0(1,1)=2;

    CudaMat<float> B1(2,2);
    B1(0,0)=7;B1(0,1)=3;
    B1(1,0)=8;B1(1,1)=4;

    blockMat.setBlockDim(2);

    blockMat.pushBackHost(A0,0,0);
    blockMat.pushBackHost(A1,1,0);
    blockMat.pushBackHost(A2,1,1);

    B.setBlockDim(2);

    B.pushBackHost(B0,0,0);
    B.pushBackHost(B1,1,0);


    blockMat.printHost();
    blockMat.getBlock(0).printHost();
    blockMat.getBlock(1).printHost();
    blockMat.getBlock(2).printHost();

    B.printHost();
    B.getBlock(0).printHost();
    B.getBlock(1).printHost();


    blockMat.host2dev();
    B.host2dev();

//    ss::ltSolve(blockMat,B);

//    B.getBlock(0).dev2host();
//    B.getBlock(0).printHost();
//    B.getBlock(1).dev2host();
//    B.getBlock(1).printHost();

    ss::ltSolve(blockMat,B,CUSPARSE_OPERATION_TRANSPOSE);

    B.getBlock(0).dev2host();
    B.getBlock(0).printHost();
    B.getBlock(1).dev2host();
    B.getBlock(1).printHost();

*/


    //thrust::device_vector<thrust::device_vector<float> > d_csrVal;



//    CudaSparseMat<float> full;
//    full.reset(10);
//    full.initHost();
//    full.pushBackHost(4 ,0,0);
//    full.pushBackHost(12 ,0,1);
//    full.pushBackHost(-16 ,0,2);
//    full.pushBackHost(12 ,1,0);
//    full.pushBackHost(37 ,1,1);
//    full.pushBackHost(-43 ,1,2);
//    full.pushBackHost(-16 ,2,0);
//    full.pushBackHost(-43 ,2,1);
//    full.pushBackHost(98 ,2,2);

//    full.printHost();
//    full.host2dev();


//    ss::choleskyDecompose(full);
//    full.dev2host();
//    full.printHost();

//    CudaMat<float> Inv;
//    Inv.initDevIdentity(full.cols());
//    ss::solve(full,Inv,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_SOLVE_POLICY_NO_LEVEL);
//    printf("inv \n");
//    Inv.dev2host();
//    Inv.printHost();

//    ss::solve(full,Inv,CUSPARSE_OPERATION_TRANSPOSE,CUSPARSE_OPERATION_NON_TRANSPOSE,CUSPARSE_SOLVE_POLICY_USE_LEVEL);
//    //ss::choleskySolve(full,Inv);
//    printf("inv \n");
//    Inv.dev2host();
//    Inv.printHost();



//    CudaSparseMat<float> L;
//    L.reset(10);
//    L.initHost();
//    L.pushBackHost(2 ,0,0);
//    L.pushBackHost(6 ,1,0);
//    L.pushBackHost(1 ,1,1);
//    L.pushBackHost(-8 ,2,0);
//    L.pushBackHost(5,2,1);
//    L.pushBackHost(3 ,2,2);

//    L.printHost();
//    L.host2dev();
////    CudaMat<float> LInv;
////    LInv.initDevIdentity(L.rows());
////    ss::choleskySolve(L,LInv);
////    LInv.dev2host();
////    LInv.printHost();


//    full.addRows(L);



    CudaSparseMat<float> sparse;


    Gpr testGP;
    size_t div=100;
    size_t samples=30;
    size_t itterations = 3;
    float upper,lower;
    upper=10;
    lower=0;
    CudaMat<float> hp(1,1);
    hp(0,0)=2;  // set the hyperparameters
    hp.host2dev();
    testGP.setHyperParam(hp);
    testGP.genPredVect2d(lower,upper,div);

    CudaMat<float> x(samples,2); // samples rows, 2 colums to represent 2d point
    CudaMat<float> xall(samples*itterations,2);
    for(size_t n = 0 ; n<itterations ; n++){
        for(size_t i = 0 ; i<samples ; i++){
            float rand1 = (std::rand()/((RAND_MAX)/10.0))+n*5;
            float rand2 = std::rand()/((RAND_MAX)/10.0);
            x(i,0) = rand1; // x point
            xall(i+samples*n,0) =rand1;
            x(i,1) = rand2; // y point
            xall(i+samples*n,1) =rand2;
        }

    }

    CudaMat<float> x2(samples,2); // samples rows, 2 colums to represent 2d point

    for(size_t i = 0 ; i<samples ; i++){
        float rand1 = std::rand()/((RAND_MAX)/10.0);
        float rand2 = std::rand()/((RAND_MAX)/10.0);
        x2(i,0) = rand1; // x point
        xall(i+samples*2,0) =rand1;
        x2(i,1) = rand2; // y point
        xall(i+samples*2,1) =rand2;
    }

   // x.printHost();
    x2.host2dev();
    x.host2dev();

    CudaMat<float> s11 = testGP.genCovariance(x,x,&gpKernel2d,hp);
    s11.dev2host();
    //.printHost();
    sparse.initFromDense(s11);


//    ss::choleskyDecompose(s11);
//    sparse.dev2host();
//    //sparse.printHost();

    ss::choleskyDecompose(sparse);
    printf("cholesky decomposition matrix \n");
    sparse.dev2host();
    sparse.printHost();


    return 0;

}
