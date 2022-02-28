#include "myQR.h"

#define TIMING

int main(int argn, char *argv[])
{
    double *A, *Q, *R, *C, *I, *Qt; // C=Q*R    I=Qt*Q
    float time;
    int M, N, flag_output; // row col
    FILE *f; //file results
    char filename[256];
    cudaEvent_t start, stop; 
    // check input argv
    if (argn < 4)
    {
        printf("%s <NumbElementsRow><NumElementColm><flagoutput> M>=N\nRunning %s 6 5\n", argv[0], argv[0]);
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
            printf("%s M(columns) N(rows) M must be greater than N\n", argv[0]);
            M = 6;
            N = 5;
            printf("Running\n%s %d %d\n", argv[0], M, N);
        }
    }

#ifdef TIMING
    sprintf(filename, "list.cpu.%d_%d", M, N); //create file results
    f = fopen(filename, "a");
#endif
    //allocation host
    A = (double *)malloc(M * N * sizeof(double));
    Q = (double *)malloc(M * N * sizeof(double));
    R = (double *)calloc(N * N, sizeof(double)); // R must be all 0
    C = (double *)calloc(M * N, sizeof(double)); // C = Q*R controllo risultato fattorizzazione
    I = (double *)calloc(N * N, sizeof(double)); // I = Q^t*Q controllo risultato
    //init A and copy in Q
    initMat(A, M, N);
    copyMat(A, Q, M, N); // A->Q

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    QRdec(A, Q, R, M, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Tempi esecuzione sul host: %8.2f ms (%3.3fs)\n", time, time / 1e3);

#ifdef TIMING
    fprintf(f, "%fms  %fs\n", time, time / 1e3); // save result in file
#endif
    //check QR decomposition
    prodMat(Q, R, C, M, N, N);
    Qt = tranMat(Q, M, N);
    prodMat(Qt, Q, I, N, M, N);

    if (flag_output)
    {
        printf("\nA\n");
        printMat(A, M, N, ACCURACY);
        printf("\nA column-major\n");
        printLinMat(A, M, N);
        printf("\nQ\n");
        printMat(Q, M, N, ACCURACY);
        printf("\nR\n");
        printMat(R, N, N, ACCURACY);
        printf("\nQ*R:\n");
        printMat(C, M, N, ACCURACY);
        printf("\nQt*Q\n");
        printMat(I, N, N, ACCURACY);
    }

    //deallocation
    free(A);
    free(Q);
    free(R);

    free(C);
    free(Qt);
    free(I);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

#ifdef TIMING
    fclose(f);
#endif

    exit(0);
}