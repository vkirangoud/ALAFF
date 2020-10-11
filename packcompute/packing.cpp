#include <stdio.h>
#include <stdlib.h>
#include "Params.h"

dim_t MC = 72;
dim_t KC = 256;
dim_t NC = 4080;

dim_t MR = 6;
dim_t NR = 8;

size_t  dgemm_pack_get_size(char identifier, const int m, const int n, const int k)
{
    size_t tbytes = 0;
    if ('A' == identifier)
    {
        // Size of single packed A buffer is MC x KC elements - row-panels of A
      // Number of elements in row-panel of A = MC x KC
      // size of micro-panels is MR x KC

        dim_t m_p_pad = ( (m + MR - 1) / MR) * MR;
        dim_t ps_n = m_p_pad * k; //  size of all packed buffer (multiples of MR x k)

        // if A is transposed - then A' dimensions will be k x m
        // here k should be multiple of MR
        dim_t mt_p_pad = ( (k + MR - 1) / MR) * MR;

        dim_t ps_t = mt_p_pad * m;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof(double);
    }
    else
    {
        // size of micro-panels is KC x NR
        dim_t n_p_pad = ( (n + NR - 1) / NR) * NR;
        dim_t ps_n = k * n_p_pad; // size of packed buffer of B (multiples of k x NR)

        // if B is transposed then B' - dimension is n x k
        // here k should be multiple of NR
        dim_t nt_p_pad = ( (k + NR - 1) / NR) * NR;
        dim_t ps_t = n * nt_p_pad;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof(double);
    }
    return tbytes;
}


size_t  sgemm_pack_get_size(char identifier, const int m, const int n, const int k)
{
    size_t tbytes = 0;
    if ('A' == identifier)
    {
        // Size of single packed A buffer is MC x KC elements - row-panels of A
      // Number of elements in row-panel of A = MC x KC
      // size of micro-panels is MR x KC

        dim_t m_p_pad = ( (m + MR - 1) / MR) * MR;
        dim_t ps_n = m_p_pad * k; //  size of all packed buffer (multiples of MR x k)

        // if A is transposed - then A' dimensions will be k x m
        // here k should be multiple of MR
        dim_t mt_p_pad = ( (k + MR - 1) / MR) * MR;

        dim_t ps_t = mt_p_pad * m;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof(float);
    }
    else
    {
        // size of micro-panels is KC x NR
        dim_t n_p_pad = ( (n + NR - 1) / NR) * NR;
        dim_t ps_n = k * n_p_pad; // size of packed buffer of B (multiples of k x NR)

        // if B is transposed then B' - dimension is n x k
        // here k should be multiple of NR
        dim_t nt_p_pad = ( (k + NR - 1) / NR) * NR;
        dim_t ps_t = n * nt_p_pad;

        // We pick the max size to ensure handling the transpose case.
        dim_t ps_max = bli_max(ps_n, ps_t);

        tbytes = ps_max * sizeof(float);
    }
    return tbytes;
}

void sgemm_pack_A(const dim_t m, const dim_t n, const dim_t k, const float alpha,
    const float* src, const dim_t ld, float* dest)
{

}


void PackMicroPanelA_MRxKC(dim_t m, dim_t k, double* A, dim_t ldA, double* Atilde)
/* Pack a micro-panel of A into buffer pointed to by Atilde.
   This is an unoptimized implementation for general MR and KC. */
{
    /* March through A in column-major order, packing into Atilde as we go. */

    if (m == MR)   /* Full row size micro-panel.*/
        for (int p = 0; p < k; p++)
            for (int i = 0; i < MR; i++)
                *Atilde++ = alpha(i, p);
    else /* Not a full row size micro-panel.  */
    {
        for (dim_t p = 0; p < k; p++) {
            for (dim_t i = 0; i < m; i++) *Atilde++ = alpha(i, p);
            for (dim_t i = m; i < MR; i++) *Atilde++ = 0.0;
        }
    }
}

void PackBlockA_MCxKC(dim_t m, dim_t k, double* A, dim_t ldA, double* Atilde)
/* Pack a MC x KC block of A.  MC is assumed to be a multiple of MR.  The block is
   packed into Atilde a micro-panel at a time. If necessary, the last micro-panel
   is padded with rows of zeroes. */
{
    for (dim_t i = 0; i < m; i += MR) {
        dim_t ib = bli_min(MR, m - i);
        PackMicroPanelA_MRxKC(ib, k, &(alpha(i, 0)), ldA, Atilde);
        //Atilde += ib * k;
        Atilde += (MR * k);
    }
}


void dgemm_pack_A(const dim_t m, const dim_t n, const dim_t k, const double alpha1,
    const double* src, const dim_t ld, double* dest)
{
    double* A = NULL;
    double* Atilde = dest;

    for (dim_t p = 0; p < k; p += KC) {
        dim_t pb = bli_min(k - p, KC);
        for (dim_t i = 0; i < m; i += MC)
        {
            dim_t ib = bli_min(m - i, MC);
            // when ib is not multiple of MR
            // At the boundary ib % MR will be filled with zeros
            A = (double*)(&(src[i * ld + p]));
            PackBlockA_MCxKC(ib, pb, A, ld, Atilde); // always fills Multiple of MR x KC
            dim_t ib_mult_MR = ((ib + MR - 1) / MR) * MR;
            Atilde += pb * ib_mult_MR;
        }
    }
}

void dumpPackBuffer(char* str, double* ap, dim_t  numbytes)
{
    FILE* fp = NULL;
    fp = fopen(str, "wt");
    if (fp == NULL)
    {
        printf("Error opening the matrix file\n");
        exit(1);
    }
    dim_t numElements = numbytes / sizeof(double);
    for (dim_t i = 0; i < numElements; i++)
    {
        fprintf(fp, "%f ", ap[i]);
    }

    fclose(fp);

    return;
}

