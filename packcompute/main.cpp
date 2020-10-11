
#include <stdio.h>
#include <stdlib.h>

#include "Params.h"


size_t  dgemm_pack_get_size(char identifier, const int m, const int n, const int k);
size_t  sgemm_pack_get_size(char identifier, const int m, const int n, const int k);

void dgemm_pack_A(const dim_t m, const dim_t n, const dim_t k, const double alpha1,
    const double* src, const dim_t ld, double* dest);

void sgemm_pack_A(const dim_t m, const dim_t n, const dim_t k, const float alpha1,
    const float* src, const dim_t ld, float* dest);

void dgemmCompute(int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc);

void dumpPackBuffer(char* str, double* ap, dim_t  numbytes);

extern dim_t MC;
extern dim_t KC;
extern dim_t NC;

int main(int argc, char** argv)
{
    FILE* fp = NULL;
    int m, k, n;
    int lda, ldb, ldc;

    double* ap = NULL;
    double* bp = NULL;
    double* cp = NULL;

    double alpha1 = 1.0f;

    double* packbuf = NULL;

    if (argc < 2)
    {
        printf("Usage: ./PackMatrix.x input.txt \n");
        exit(1);
    }

    fp = fopen(argv[1], "r");
    if (fp == NULL)
    {
        printf("Error opening the input file %s \n", argv[1]);
        exit(1);
    }
    char filename[100];
    printf("MC = %ld KC = %ld NC = %ld MR = %ld NR = %ld\n", MC, KC, NC, MR, NR);

    while (fscanf(fp, "%d %d %d %d %d %d\n", &m, &k, &n, &lda, &ldb, &ldc) == 6) {
        // Row major
        int nelems_A = (m - 1) * lda + (k - 1) * 1 + 1; // rs_a = lda, cs = 1
        int nelems_B = (k - 1) * ldb + (n - 1) * 1 + 1; // rs_b = ldb, cs = 1
        int nelems_C = (m - 1) * ldc + (n - 1) * 1 + 1; // rs_c = ldb, cs = 1

        // column-major
       // int nelems_A = (m - 1) * 1 + (k - 1) * lda + 1; // rs_a = 1, cs = lda
       // int nelems_B = (k - 1) * 1 + (n - 1) * ldb + 1; // rs_b = 1, cs = ldb

        ap = (double*)malloc(sizeof(double) * nelems_A);
        if (ap == NULL) { printf("Error allocation memory A \n"); exit(1); }

        bp = (double*)malloc(sizeof(double) * nelems_B);
        if (bp == NULL) { printf("Error allocation memory B \n"); exit(1); }

        cp = (double*)malloc(sizeof(double) * nelems_C);
        if (cp == NULL) { printf("Error allocation memory C \n"); exit(1); }
        for (int i = 0; i < nelems_C; i++) cp[i] = 0.0;

        double* cp_ref = (double*)malloc(sizeof(double) * nelems_C);
        if (cp_ref == NULL) { printf("Error allocation memory C \n"); exit(1); }
        for (int i = 0; i < nelems_C; i++) cp_ref[i] = 0;

        // create a matrix row-major
        gen_random_matrix(ap, m, k, lda, 1);
        gen_random_matrix(bp, k, n, ldb, 1);
//        gen_random_matrix(cp_ref, m, n, ldc, 1);

        MyGemm_ref(m, n, k, ap, lda, bp, ldb, cp_ref, ldc);

        // Column-major matrices
        //gen_random_matrix(ap, m, k, 1, lda); 
        //gen_random_matrix(bp, k, n, 1, ldb);

        // Compute the size of packed buffer
        dim_t totlBytes = dgemm_pack_get_size('A', m, n, k);

        // allocate memory to packed buffer
        packbuf = (double*)malloc(totlBytes);
        if (packbuf == NULL)
        {
            printf("Error allocating memory \n");
            exit(1);
        }
        dim_t elements = totlBytes / sizeof(double);
        for (dim_t i = 0; i < elements; i++) packbuf[i] = 0.0;

        // Perform packing of A
        dgemm_pack_A(m, n, k, alpha1, ap, lda, packbuf);
        
       // sprintf(filename, "packA%d_%d_%d.txt", m, n, k);
        // Dump to a file
       // dumpPackBuffer(filename, packbuf, totlBytes);

        // Perform matrix multiplication
        //void dgemmCompute(int m, int n, int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
        dgemmCompute(m, n, k, packbuf, lda, bp, ldb, cp, ldc);

        if (isMatrixMatch(cp_ref, cp, m, n, ldc) == 1)
        {
            printf(" %d %d %d %d %d %d ---> Passed\n", m, n, k, lda, ldb, ldc);
        }
        else
        {
            printf(" %d %d %d %d %d %d ---> Failed\n", m, n, k, lda, ldb, ldc);
        }

        free(ap);
        free(bp);
        free(cp);
        free(packbuf);
    }

    return 0;
}