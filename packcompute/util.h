#pragma once
#include <stdio.h>

inline void dumpMatrix(char* str, double* A, int rows, int cols, int rs, int cs)
{
    FILE* fin = NULL;
    long i, j;
    fin = fopen(str, "wt");
    if (fin == NULL) {
        printf("Error opening the dump file\n");
        return;
    }

    for (j = 0; j < cols; j++)
    {
        for (i = 0; i < rows; i++)
        {
            fprintf(fin, "%f ", A[i * rs + j * cs]);
        }
        fprintf(fin, "\n");
    }
}



