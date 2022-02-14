#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "myQR.h"

int main(int argn, char *argv[])
{
    float *A, *Q, *R, *C, *I, *Qt;
    int M, N, flag_output; // row col
    if (argn < 4)
    {
        M = 6;
        N = 5;
        flag_output = 0;
    }
    else
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        flag_output = atoi(argv[3]);
        if ((M < N || flag_output != 0) && (M < N || flag_output != 1))
        {
            printf("./qr.out M(numero righe) N(numero colonne) M>=N\n./qr.out 6 5");
            M = 6;
            N = 5;
        }
    }
    A = (float *)malloc(M * N * sizeof(float));
    Q = (float *)malloc(M * N * sizeof(float));
    R = (float *)calloc(N * N, sizeof(float));

    C = (float *)calloc(M * N, sizeof(float)); // C = Q*R controllo risultato fattorizzazione
    I = (float *)calloc(N * N, sizeof(float)); // I = Q^t*Q controllo risultato
    initMat(A, M, N);
    copyMat(A, Q, M, N); // A->Q

    QRdec(A, Q, R, M, N);
    prodMat(Q, R, C, M, N, N);
    Qt = tranMat(Q, M, N);
    prodMat(Qt, Q, I, N, M, N);

    if (flag_output)
    {
        printf("\nA\n");
        printMat(A, M, N, ACCURACY);
        printf("\nQ\n");
        printMat(Q, M, N, ACCURACY);
        printf("\nR\n");
        printMat(R, N, N, ACCURACY);
        printf("\nQ*R:\n");
        printMat(C, M, N, ACCURACY);
        printf("\nQt*Q\n");
        printMat(I, N, N, ACCURACY);
    }

    free(A);
    free(Q);
    free(R);

    free(C);
    free(Qt);
    free(I);

    exit(0);
}