#include <stdio.h>
#include <iostream>
#include "cuda/foo.cuh"

// extern "C"
// void useCUDA();

int main()
{
    std::cout<<"Hello C++"<<std::endl;
    useCUDA();
    return 0;
}
