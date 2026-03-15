#include <iostream>
#include <chrono>
#include "matmul.h"

int main()
{

    int N = 1024;

    Matrix A(N,N);
    Matrix B(N,N);
    Matrix C(N,N);


    for(int i=0;i<N;i++)
    for(int j=0;j<N;j++)
    {
        A(i,j)=1.0;
        B(i,j)=1.0;
    }

    auto start = std::chrono::high_resolution_clock::now();

    FastMatmul::multiply(A,B,C);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;

    std::cout<<"Time "<<diff.count()<<std::endl;

}