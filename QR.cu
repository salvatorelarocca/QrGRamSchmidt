#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#define ACCURACY 2

void initMat(double *matrix, int M, int N);
void printLinMat(double *mat, int m, int n);
void printMat(double *matrix, int M, int N, int acc);
void QRdec(double *A, double *Q, double *R, int M, int N);
void copyMat(double *X, double *Y, int M, int N);
void prodMat(double *A, double *B, double *C, int m, int n, int p);
double *tranMat(double *mat, int m, int n);
void difMat(double *A, double *B, int m, int n, int acc);

int main(int argn, char *argv[])
{
    double *A, *Q, *R, *C, *I, *Qt;
    int M, N; //row col
    if (argn < 3)
    {
        M = 6;
        N = 5;
    }
    else
    {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        if (M < N)
        {
            printf("./qr.out M(numero righe) N(numero colonne) M>=N\n./qr.out 6 5");
            M = 6;
            N = 5;
        }
    }
    A = (double *)malloc(M * N * sizeof(double));
    Q = (double *)malloc(M * N * sizeof(double));
    R = (double *)calloc(N * N, sizeof(double));

    C = (double *)calloc(M * N, sizeof(double)); //C = Q*R controllo risultato fattorizzazione
    I = (double *)calloc(N * N, sizeof(double)); //I = Q^t*Q controllo risultato

    initMat(A, M, N);
    copyMat(A, Q, M, N); //A->Q

    QRdec(A, Q, R, M, N);
    printf("\nQ\n");
    printMat(Q, M, N, ACCURACY);

    /*printf("\nR\n");
    printMat(R, N, N, ACCURACY);
    prodMat(Q, R, C, M, N, N);
    printf("\nQ*R:\n");
    printMat(C, M, N, ACCURACY);
    printf("\ndiff:\n");
    difMat(A, C, M, N, ACCURACY);*/

    Qt = tranMat(Q, M, N);

    printf("\nQt\n");
    printMat(Qt, N, M, ACCURACY);
   
    printf("\nQt*Q\n");
    prodMat(Qt, Q, I, N, M, N);
    printMat(I, N, N, ACCURACY);
    
    

    free(A);
    free(Q);
    free(R);

    //free(C);
    free(Qt);
    free(I);

    exit(0);
}

/*More precisely, the first column is normalized, and its
projection onto all remaining columns is subtracted from each of them. By this, the
first column is normal, and orthogonal to all other columns. This step we repeat for
the second column, leaving the first untouched. Since the first and second column are
orthonormal, and we subtracted their projections from all the m − 2 other columns,
they are both orthogonal to all the other columns.
This procedure is repeated until all
columns of A are orthonormal, i.e. until A has been transformed into an orthonormal
matrix Q. R is obtained as a by-product from the projection coefficients. Figure 1
illustrates the computation of the i-th column of Q and the i-th row of R.

La i-esima riga di R è data dai suoi elementi r_ij che sono il prodotto
scalare tra la i-esima colonna e la j-esima colonna di Q. Le colonne di Q prima della i-esima
sono già ortonormali.

*/
void QRdec(double *A, double *Q, double *R, int M, int N)
{
    int i, j, k;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
            for (k = 0; k < M; k++)
                R[i * N + j] += (Q[k * N + i] * Q[k * N + j]);
        R[i * N + i] = sqrt(R[i * N + i]);

        for (k = 0; k < M; k++)
            Q[k * N + i] = Q[k * N + i] / R[i * N + i];

        for (j = i + 1; j < N; j++)
            R[i * N + j] = R[i * N + j] / R[i * N + i];

        for (j = i + 1; j < N; j++)
            for (k = 0; k < M; k++)
                Q[k * N + j] = Q[k * N + j] - Q[k * N + i] * R[i * N + j];
    }
}

void initMat(double *matrix, int M, int N)
{
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            //matrix[i * N + j] = ((double)rand() / (RAND_MAX)); //casuali intervallo [0 - 1]
            matrix[i * N + j] = i * N + j;
}

void copyMat(double *X, double *Y, int M, int N)
{
    int i, j;
    for (i = 0; i < M; i++)
        for (j = 0; j < N; j++)
            Y[i * N + j] = X[i * N + j];
}

void printMat(double *matrix, int M, int N, int acc)
{
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%.*f ", acc, matrix[i * N + j]);
        printf("\n");
    }
    printf("linearizzata:");
    printLinMat(matrix, M, N);
}

void prodMat(double *A, double *B, double *C, int m, int n, int p)
{
    int i, j, k;
    for (i = 0; i < m; i++)
        for (j = 0; j < p; j++)
        {
            for (k = 0; k < n; k++)
            {
                C[i * p + j] += A[i * n + k] * B[k * p + j];
            }
        }
}

double *tranMat(double *mat, int m, int n)
{
    int i, j;
    double *matT = (double *)calloc(n * m, sizeof(double));
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            matT[j * m + i] = mat[i * n + j];
        }
    }
    return matT;
}

void difMat(double *A, double *B, int m, int n, int acc)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            //printf("%.*f ", acc, fabs(A[i * n + j] - B[i * n + j]));
            printf("%d ", fabs(A[i * n + j] - B[i * n + j]) < 1e-15);
        }
        printf("\n");
    }
}

void printLinMat(double *mat, int m, int n){
    int i;
    for(i = 0; i <n*m; i++)
        printf("%.2f ", mat[i]);
}