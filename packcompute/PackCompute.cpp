#include "Params.h"

#ifdef AOCL_COLMAJOR
#define Am( i,j ) A[ (j)*lda + (i) ]   // map alpha( i,j ) to array A
#else
#define Am( i,j ) A[ (i)*lda + (j) ]   // map Am( i,j ) to array A
#define Cm(i, j ) C[ (i)*ldc + (j) ]   // map Cm( i,j ) to array C
#define Bm(i, j ) B[ (i)*ldb + (j) ]   // map Bm( i,j ) to array B

static inline double* getMatptr(double* X, int i, int j, int lda) { return &X[ (i)*lda + (j)]; }
#endif

static void dgemmMRxNRKernelPackA(int m, int n, int k, double* a, double* B, int ldb, double* C, int ldc)
{
    double* apack = a;
    for (int p = 0; p < k; p++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = 0; i < m; i++)
            {
                Cm(i, j) += apack[i] * Bm(p, j);
            }
        }
        apack += MR;
    }
}

static void dgemmMacroKernelCompute(int m, int n, int k, double* ap, double* b, int ldb, double* c, int ldc)
{
    for (int jr = 0; jr < n; jr += NR)
    {
        int jb = bli_min(n - jr, NR);
        double* pb = getMatptr(b, 0, jr, ldb); // &b(0,jr)
        double* pc = getMatptr(c, 0, jr, ldc); // &c(0,jr)

        for (int ir = 0; ir < m; ir += MR)
        {
            int ib = bli_min(m - ir, MR);
            double* pci = getMatptr(pc, ir, 0, ldc); // c(ir, jr)
            dgemmMRxNRKernelPackA(ib, jb, k, &ap[ir * k], pb, ldb, pci, ldc);
        }
    }
}


void dgemmCompute(int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{
    for (int j = 0; j < n; j += NC)
    {
        int jb = bli_min(n - j, NC);

        double* pCj = &Cm(0, j);
        double* pBj = &Bm(0, j);
        double* Apack = A;
        for (int p = 0; p < k; p += KC)
        {
            int pb = bli_min(k - p, KC);
            double* pBkj = getMatptr(pBj, p, 0, ldb);
            for (int i = 0; i < m; i += MC)
            {
                int ib = bli_min(m - i, MC);
                double* pCij = getMatptr(pCj, i, 0, ldc); // &C(i,j)
                dgemmMacroKernelCompute(ib, jb, pb, Apack, pBkj, ldb, pCij, ldc);
                int ibalign = ( (ib + MR - 1) / MR) * MR;
                Apack += ibalign * pb;
            }
        }
    }
}