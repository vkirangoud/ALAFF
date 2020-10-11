#pragma once

#include <cstdlib> // for rand() and srand()
#include <ctime> // for time()
#include <stdio.h>

typedef size_t dim_t;

extern dim_t MC;
extern dim_t KC;
extern dim_t NC;
               ;
extern dim_t MR;
extern dim_t NR;

#define bli_min( a, b )  ( (a) < (b) ? (a) : (b) )
#define bli_max( a, b )  ( (a) > (b) ? (a) : (b) )
#define bli_abs( a )     ( (a) <= 0 ? -(a) : (a) )



#ifdef AOCL_COLMAJOR
#define alpha( i,j ) A[ (j)*ldA + (i) ]   // map alpha( i,j ) to array A 
#define beta( i,j )  B[ (j)*ldB + (i) ]   // map beta( i,j )  to array B
#define gamma( i,j ) C[ (j)*ldC + (i) ]   // map gamma( i,j ) to array C

#else
#define alpha( i,j ) A[ (i)*ldA + (j) ]   // map alpha( i,j ) to array A
#define beta( i,j )  B[ (i)*ldB + (j) ]   // map beta( i,j )  to array B
#define gamma( i,j ) C[ (i)*ldC + (j) ]   // map gamma( i,j ) to array C

#endif


inline void MyGemm_ref(int m, int n, int k, double* A, int ldA,
    double* B, int ldB, double* C, int ldC)
{
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
                gamma(i, j) += alpha(i, p) * beta(p, j);
}

// Generate a random number between min and max (inclusive)
// Assumes srand() has already been called
inline int genRandomNumber(int min, int max)
{
    static const double fraction = 1.0 / (static_cast<double>(RAND_MAX) + 1.0);
    return static_cast<int>(rand() * fraction * (max - min + 1) + min);
}

inline void gen_random_matrix(double* A, int rows, int cols, int rs, int cs)
{
    long i, j;
    //unsigned short xsubi[3] = { 918, 729, 123 };
    srand(static_cast<unsigned int>(time(0))); // set initial seed value to system clock

    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            A[i * rs + j * cs] = (double)genRandomNumber(1, 3); // 1 + nrand48(xsubi) % 9;
        }
}

inline int isMatrixMatch(double* ref, double* mat, int rows, int cols, int ldc)
{
    long i, j;
    for (j = 0; j < cols; j++)
    {
        for (i = 0; i < rows; i++)
        {
            #ifdef AOCL_COLMAJOR
            if (ref[i + j * ldc] != mat[i + j * ldc])
#else
            if (ref[i * ldc + j ] != mat[i * ldc + j ])
#endif
            {
                printf("Mismatch at (%d, %d)\n", i, j);
                return 0;
            }
        }
    }
    return 1;
}






