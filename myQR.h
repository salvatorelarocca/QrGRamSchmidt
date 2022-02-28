#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#define ACCURACY 5 //accuratezza nelle stampe
#define IDX(i, j, ld) ((j * ld) + i) //le matrici sono allocate linearmente per colonna

void initMat(double *matrix, int M, int N);
void printLinMat(double *mat, int m, int n);
void printMat(double *matrix, int M, int N, int acc);
void QRdec(double *A, double *Q, double *R, int M, int N);
void copyMat(double *X, double *Y, int M, int N);
void prodMat(double *A, double *B, double *C, int m, int n, int p);
double *tranMat(double *mat, int m, int n);
void difMat(double *A, double *B, int m, int n, int acc);


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
                R[IDX(i, j, N)] += Q[IDX(k, i, M)] * Q[IDX(k, j, M)];
        R[IDX(i, i, N)] = sqrt(R[IDX(i, i, N)]);

        for (k = 0; k < M; k++)
            Q[IDX(k, i, M)] = Q[IDX(k, i, M)] / R[IDX(i, i, N)];

        for (j = i + 1; j < N; j++)
            R[IDX(i, j, N)] = R[IDX(i, j, N)] / R[IDX(i, i, N)];

        for (j = i + 1; j < N; j++)
            for (k = 0; k < M; k++){
                Q[IDX(k, j, M)] = Q[IDX(k, j, M)] - Q[IDX(k, i, M)] * R[IDX(i, j, N)];
            }
    }
}
void initMat(double *matrix, int M, int N)
{
    int i;
    //srand(time(NULL));
    for (i = 0; i < M * N; i++)
        matrix[i] = ((double)rand() / (RAND_MAX)); // casuali intervallo [0 - 1]
}


void copyMat(double *X, double *Y, int M, int N)
{
    int i;
    for (i = 0; i < M * N; i++)
        Y[i] = X[i];
}

void printMat(double *matrix, int M, int N, int acc)
{
    int i, j;
    for (i = 0; i < M; i++)
    {
        for (j = 0; j < N; j++)
            printf("%.*f ", acc, matrix[IDX(i,j,M)]);
        printf("\n");
    }
    /*printf(" linearizzata:");
    printLinMat(matrix, M, N);*/
}

void prodMat(double *A, double *B, double *C, int m, int n, int p)
{
    int i, j, k;
    for (i = 0; i < m; i++)
        for (j = 0; j < p; j++)
        {
            for (k = 0; k < n; k++)
            {
                C[IDX(i,j,m)] += A[IDX(i,k,m)] * B[IDX(k,j,n)];
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
            matT[IDX(j,i,n)] = mat[IDX(i,j,m)];
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
            // printf("%.*f ", acc, fabs(A[i * n + j] - B[i * n + j]));
            printf("%d ", fabs(A[i * n + j] - B[i * n + j]) < 1e-15);
        }
        printf("\n");
    }
}

void printLinMat(double *mat, int m, int n)
{
    int i;
    for (i = 0; i < n * m; i++)
        printf("%.2f ", mat[i]);
}