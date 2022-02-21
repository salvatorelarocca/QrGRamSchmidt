#include <cuda.h>
#include "myQR.h"

#define BLOCK1 4

#define BLOCK1X 8 // deve essere multiplo di 2
#define BLOCK1Y 2

#define BLOCK2X 10
#define BLOCK2Y 2

__global__ void mult(float *Q, float *R, int m, int n, int k);
__global__ void scaleR(float *Q, float *R, int m, int n, int k, float S);
__global__ void scaleQ(float *Q, float *R, int m, int n, int k, float S);
__global__ void update(float *Q, float *R, int m, int n, int k);
__global__ void step3(float *A, float *R, int M, int N, int k);
int QR(float *Q, float *R, unsigned int m, unsigned int n);

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

  QR(Q, R, M, N);

  //check ortonormalità
  Qt = tranMat(Q, M, N);
  prodMat(Qt, Q, I, N, M, N);

  printf("\nA\n");
  printMat(A, M, N, ACCURACY);
  printf("\nQ\n");
  printMat(Q, M, N, ACCURACY);
  printf("\nR\n");
  printMat(R, N, N, ACCURACY);
  printf("\nI\n");
  printMat(I, N, N, ACCURACY);
  

  free(A);
  free(Q);
  free(R);
  free(C);
  free(I);
}

__global__ void mult(float *Q, float *R, int m, int n, int k)
{
  __shared__ float RS[BLOCK1Y][BLOCK1X];

  int tid1 = threadIdx.x;
  int tid2 = threadIdx.y;
  int i = blockIdx.x * BLOCK1Y + tid2 + k;

  float S = 0.0f;
  // printf("block:%d(tx:%d, ty:%d)\n", blockIdx.x, tid1, tid2);
  if (i < k || i >= n)
  {
    // printf("(tx:%d, ty:%d)\n", tid1, tid2);
    return;
  }

  for (int j = tid1; j < m; j += BLOCK1X)
  {
    S += Q[IDX(j, k, m)] * Q[IDX(j, i, m)];
    // printf("block %d (%.2f * %.2f) i:%d, j:%d (tx:%d, ty:%d)\n", blockIdx.x, Q[IDX(j,k,m)], Q[IDX(j,i,m)], i, j, tid1, tid2);
  }

  // thread writes result in shared array RS

  RS[tid2][tid1] = S;

  // printf("block %d, mem[%d][%d]=%.2f\n", blockIdx.x, tid2, tid1, S);

  int NT = BLOCK1X;

  while (NT > 1)
  {
    // first half of threads sums up
    __syncthreads();
    NT = NT >> 1;
    if (tid1 < NT)
    { 
      // provare se possibile a fare la radice di rkk all'interno di questo kernel
      // printf("%d\n", NT);
      // printf("(%d, %d) %.2f + %.2f\n", tid1, tid2, RS[tid2][tid1], RS[tid2][tid1+NT]);
      RS[tid2][tid1] += RS[tid2][tid1 + NT];
    }
  }

  // now thread 0 writes the result

  if (tid1 == 0)
  {
    R[IDX(k, i, n)] = RS[tid2][0];
    //printf("block:%d thread(%d, %d), write result: %.2f in R[%d][%d]\n", blockIdx.x, tid1, tid2, R[IDX(k,i,n)], k, i);
  }
}

__global__ void scaleR(float *Q, float *R, int m, int n, int k, float S)
{
  int i = blockIdx.x * BLOCK1 + threadIdx.x + k;
  //printf("block:%d tx:%d, i:%d\n\n", blockIdx.x, threadIdx.x, i);
  if (i >= k && i < n)
  {
    //printf("block:%d tx:%d, i:%d\n", blockIdx.x, threadIdx.x, i);
    R[IDX(k, i, n)] *= S;
    //printf("R[%d][%d]: %.2f\n", k, i, R[IDX(k, i, n)]);
  }
}

__global__ void scaleQ(float *Q, float *R, int m, int n, int k, float S)
{
  int i = blockIdx.x * BLOCK1 + threadIdx.x; //+1
  //printf("block:%d tx:%d, i:%d\n", blockIdx.x, threadIdx.x, i);
  if (i < m)
  {
    Q[IDX(i, k, m)] *= S;
    //printf("Q[%d][%d] = %.2f \n", i, k, Q[IDX(i, k, m)]);
  }
}

// orthonormalization secondo il paper
/*__global__ void update(float *Q, float *R, int m, int n, int k)
{
  int tid1 = threadIdx.x;
  int tid2 = threadIdx.y;
  int j = blockIdx.y * BLOCK2Y + tid2 + k + 1;
  if (j < k + 1 || j > n) return;
  for (int i = tid1 + 1; i < m; i += BLOCK2X)
  {
    Q[IDX(i, j, m)] -= Q[IDX(i, k, m)] * R[IDX(k, j, n)];
  }
}*/

__global__ void step3(float *A, float *R, int M, int N, int k)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;         // row index
    int j = threadIdx.x + blockDim.x * blockIdx.x + k + 1; // col index + k + 1
    if (i < M && j < N)
    {
        // printf("(Blk.x:%d Blk.y:%d) (tx:%d ty:%d) access(i:%d, j:%d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
        //printf("%0.f -  %.0f * %.0f\n", A[IDX(i, j, M)], A[IDX(i, k, M)], R[IDX(k, j, N)]);
        A[IDX(i, j, M)] -= A[IDX(i, k, M)] * R[IDX(k, j, N)];
    }
}

int QR(float *Q, float *R, unsigned int m, unsigned int n)
{
  float *QGPU; // Q on GPU
  float *RGPU; // R on GPU
  // Allocate on GPU
  cudaMalloc((void **)&QGPU, m * n * sizeof(float));
  cudaMalloc((void **)&RGPU, n * n * sizeof(float));

  cudaMemcpy(QGPU, Q, m * n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(RGPU, R, n * n * sizeof(float), cudaMemcpyHostToDevice);

  dim3 dimGrid;
  dim3 dimBlock;

  for (unsigned int k = 0; k < n; k++)
  {
    //printf("k: %d\n", k);
    dimGrid = dim3((n - k + BLOCK1Y) / BLOCK1Y, 1, 1);
    dimBlock = dim3(BLOCK1X, BLOCK1Y, 1);
    //printf("\nkernel mult\ndimgrid x: %d\n", (n - k + BLOCK1Y) / BLOCK1Y);
    //printf("liv_kernel mult dimgrid x: %d\n", (n - k) / BLOCK1Y + (((n-k) % BLOCK1Y) == 0 ? 0 : 1));

    mult<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);

    // fattore di scala
    float S;
    cudaMemcpy(&S, &RGPU[IDX(k, k, n)], sizeof(float), cudaMemcpyDeviceToHost);
    S = sqrt(S);
    S = 1.0 / S;
    //printf("1/S = %.2f\n", S);

    dimGrid = dim3((m + BLOCK1 - 1) / BLOCK1, 1, 1);
    dimBlock = dim3(BLOCK1, 1, 1);
    /*printf("\nkernel scaleQ\n");
    printf("dimgrid x: %d\n", (m + BLOCK1 - 1) / BLOCK1);
    printf("liv_dimgrid x: %d\n", m / BLOCK1 + ((m % BLOCK1) == 0 ? 0 : 1));*/
    scaleQ<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k, S);

    dimGrid = dim3((n - k + BLOCK1) / BLOCK1, 1, 1);
    dimBlock = dim3(BLOCK1, 1, 1);
    /*printf("\nkernel scaleR\n");
    printf("dimgrid x: %d\n", (n - k + BLOCK1) / BLOCK1);
    printf("liv_dimgrid x: %d\n", (n - k) / BLOCK1 + (((n - k) % BLOCK1) == 0 ? 0 : 1));*/
    scaleR<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k, S);

    //Terzo step secondo il paper
    /*printf("dimgrid y: %d\n", (n - k + BLOCK2Y) / BLOCK2Y);
    printf("liv_dimgrid y: %d\n", (n - k) / BLOCK2Y + (((n - k) % BLOCK2Y) == 0 ? 0 : 1));
    dimGrid = dim3(1, (n - k + BLOCK2Y) / BLOCK2Y, 1);
    dimBlock = dim3(BLOCK2X, BLOCK2Y, 1);
    update<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);*/

    //Terzo step secondo me
    dimGrid = dim3((n-k)/BLOCK1X + (((n - k) % BLOCK1X) == 0 ? 0 : 1), m/BLOCK1Y + ((m % BLOCK1Y) == 0 ? 0 : 1), 1);
    dimBlock = dim3(BLOCK1X, BLOCK1Y, 1);
    step3<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);
  }

  cudaMemcpy(Q, QGPU, m * n * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(R, RGPU, n * n * sizeof(float), cudaMemcpyDeviceToHost);

  cudaFree(QGPU);
  cudaFree(RGPU);

  return 0;
}