#include <cuda.h>
#include "myQR.h"

#define DEBUG
#define TIMING

#define BLOCK1X 8 // deve essere multiplo di 2
#define BLOCK1Y 2
#define BLOCK1 2

__global__ void mult(double *Q, double *R, int m, int n, int k);
__global__ void scaleR(double *Q, double *R, int m, int n, int k, double S);
__global__ void scaleQ(double *Q, double *R, int m, int n, int k, double S);
__global__ void update(double *Q, double *R, int m, int n, int k);
__global__ void step3(double *A, double *R, int M, int N, int k);
int QR(double *Q, double *R, unsigned int m, unsigned int n, int blockx, int blocky, int blocknorm);

int main(int argn, char *argv[])
{
  double *A, *Q, *R, *C, *I, *Qt; // C = Q*R  I = Qt*Q check results
  int M, N, flag_output;          // row col flagResults
  int blockx, blocky, block_norm; // blockx, blocky dimension for first(mult) and third(step3) kernel. block_norm dimension for kernel ScaleR and ScaleQ
  if (argn < 7)
  {
    M = 7;
    N = 5;
    blockx = 8;
    blocky = 2;
    block_norm = 4;
    printf("%s <NumbElementsRow><NumElementsColm><NumbThreadsBlockX><NumbThreadsBlockY><NumbThreadsNorm><flagoutput> M>=N\n%s (%d, %d) (%d, %d) %d\n", argv[0], argv[0], M, N, blockx, blocky, block_norm);
    flag_output = 1;
  }
  else
  {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    blockx = atoi(argv[3]);
    blocky = atoi(argv[4]);
    block_norm = atoi(argv[5]);
    flag_output = atoi(argv[6]);
    if ((M < N || flag_output != 0) && (M < N || flag_output != 1))
    {
      M = 7;
      N = 5;
      flag_output = 1;
      printf("%s <NumbElementsRow><NumElementsColm><NumbThreadsBlockX><NumbThreadsBlockY><NumbThreadsNorm><flagoutput> M>=N\n%s (%d, %d) (%d, %d) %d\n", argv[0], argv[0], M, N, blockx, blocky, block_norm);
    }
  }

  A = (double *)malloc(M * N * sizeof(double));
  Q = (double *)malloc(M * N * sizeof(double));
  R = (double *)calloc(N * N, sizeof(double));
  C = (double *)calloc(M * N, sizeof(double)); // C = Q*R controllo risultato fattorizzazione
  I = (double *)calloc(N * N, sizeof(double)); // I = Q^t*Q controllo risultato
#ifdef DEBUG
  printf("allocated A Q R C I\n");
#endif
  initMat(A, M, N);
  copyMat(A, Q, M, N); // A->Q
#ifdef DEBUG
  printf("initialized A\ncopy A in Q\n");
#endif

  QR(Q, R, M, N, blockx, blocky, block_norm);
#ifdef DEBUG
  printf("QR decomposition in cuda done \n");
#endif

  if (flag_output)
  {
#ifdef DEBUG
    printf("check results\n");
#endif
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
  }

#ifdef DEBUG
  printf("\ndeallocation\n");
#endif
  free(A);
  free(Q);
  free(R);
  free(C);
  free(I);
#ifdef DEBUG
  printf("\nend deallocation\n");
#endif
}

__global__ void mult(double *Q, double *R, int m, int n, int k)
{
  //__shared__ double RS[BLOCK1Y][BLOCK1X];
  extern __shared__ double rs[]; // rs[blocky][blockx]
  int tid1 = threadIdx.x;
  int tid2 = threadIdx.y;
  int i = blockIdx.x * blockDim.y + tid2 + k;

  double S = 0.0f;
  // printf("block:%d(tx:%d, ty:%d)\n", blockIdx.x, tid1, tid2);
  if (i < k || i >= n)
  {
    // printf("(tx:%d, ty:%d)\n", tid1, tid2);
    return;
  }

  for (int j = tid1; j < m; j += blockDim.x)
  {
    S += Q[IDX(j, k, m)] * Q[IDX(j, i, m)];
    // printf("block %d (%.2f * %.2f) i:%d, j:%d (tx:%d, ty:%d)\n", blockIdx.x, Q[IDX(j,k,m)], Q[IDX(j,i,m)], i, j, tid1, tid2);
  }

  // thread writes result in shared array RS
  int index = tid2 * blockDim.x + tid1;
  rs[index] = S;
  // RS[tid2][tid1] = S;
  // printf("block %d, mem[%d][%d]=%.2f r[%d]=%.2f\n", blockIdx.x, tid2, tid1, S, index, r[index]);

  int NT = blockDim.x;

  while (NT > 1)
  {
    // first half of threads sums up
    __syncthreads();
    NT = NT >> 1;
    if (tid1 < NT)
    {
      // printf("RS[%d][%d]=%.2f + RS[%d][%d]=%.2f -- r[%d]=%.2f + r[%d]=%.2f\n\n", tid2, tid1, RS[tid2][tid1], tid2, tid1 + NT, RS[tid2][tid1 + NT], index, r[index], index + NT, r[index + NT]);
      // RS[tid2][tid1] += RS[tid2][tid1 + NT];
      rs[index] += rs[index + NT];
    }
  }

  // now thread 0 writes the result

  if (tid1 == 0)
  {
    // R[IDX(k, i, n)] = RS[tid2][0];
    // printf("%.2f ", R[IDX(k, i, n)]);
    R[IDX(k, i, n)] = rs[index];
    // printf("%.2f ", R[IDX(k, i, n)]);
    // printf("block:%d thread(%d, %d), write result: %.2f in R[%d][%d]\n", blockIdx.x, tid1, tid2, R[IDX(k,i,n)], k, i);
  }
}

__global__ void scaleR(double *Q, double *R, int m, int n, int k, double S)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + k;
  // printf("block:%d tx:%d, i:%d\n\n", blockIdx.x, threadIdx.x, i);
  if (i >= k && i < n)
  {
    // printf("block:%d tx:%d, i:%d\n", blockIdx.x, threadIdx.x, i);
    R[IDX(k, i, n)] *= S;
    // printf("R[%d][%d]: %.2f\n", k, i, R[IDX(k, i, n)]);
  }
}

__global__ void scaleQ(double *Q, double *R, int m, int n, int k, double S)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //+1
  // printf("block:%d tx:%d, i:%d\n", blockIdx.x, threadIdx.x, i);
  if (i < m)
  {
    Q[IDX(i, k, m)] *= S;
    // printf("Q[%d][%d] = %.2f \n", i, k, Q[IDX(i, k, m)]);
  }
}

// orthonormalization secondo il paper
/*__global__ void update(double *Q, double *R, int m, int n, int k)
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

__global__ void step3(double *A, double *R, int M, int N, int k)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;         // row index
  int j = threadIdx.x + blockDim.x * blockIdx.x + k + 1; // col index + k + 1
  if (i < M && j < N)
  {
    // printf("(Blk.x:%d Blk.y:%d) (tx:%d ty:%d) access(i:%d, j:%d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
    // printf("%0.f -  %.0f * %.0f\n", A[IDX(i, j, M)], A[IDX(i, k, M)], R[IDX(k, j, N)]);
    A[IDX(i, j, M)] -= A[IDX(i, k, M)] * R[IDX(k, j, N)];
  }
}

int QR(double *Q, double *R, unsigned int m, unsigned int n, int blockx, int blocky, int blocknorm)
{
  double *QGPU; // Q on GPU
  double *RGPU; // R on GPU
  float time;
  dim3 dimGrid, dimGridScaleQ;
  dim3 dimBlock, dimBlockScaleQ;
  int shMem = blockx * blockx * sizeof(double);
  // Constant dimGrid and dimBlock for kernel ScaleQ
  dimGridScaleQ = dim3((m + blocknorm - 1) / blocknorm, 1, 1);
  dimBlockScaleQ = dim3(blocknorm, 1, 1);
  cudaEvent_t start, stop;
#ifdef TIMING
  FILE *f; // file results
  char filename[256];
  sprintf(filename, "list.gpu.%d_%d", m, n); // create file results
  f = fopen(filename, "a");
#endif

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate on GPU
  cudaEventRecord(start, 0);
  cudaMalloc((void **)&QGPU, m * n * sizeof(double));
  cudaMalloc((void **)&RGPU, n * n * sizeof(double));

  cudaMemcpy(QGPU, Q, m * n * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(RGPU, R, n * n * sizeof(double), cudaMemcpyHostToDevice);

  for (unsigned int k = 0; k < n; k++)
  {
    dimGrid = dim3((n - k + blocky) / blocky, 1, 1);
    dimGrid = dim3((n - k) / blocky + (((n - k) % blocky) == 0 ? 0 : 1));
    dimBlock = dim3(blockx, blocky, 1);
    // printf("\nkernel mult with k:%d\ndimgrid x:%d, y:%d, z:%d\n", k, dimGrid.x, dimGrid.y, dimGrid.z);
    mult<<<dimGrid, dimBlock, shMem>>>(QGPU, RGPU, m, n, k);

    // fattore di scala
    double S;
    cudaMemcpy(&S, &RGPU[IDX(k, k, n)], sizeof(double), cudaMemcpyDeviceToHost);
    S = sqrt(S);
    S = 1.0 / S;

    // printf("\nkernel scaleQ\n");
    // printf("dimgrid x: %d\n", (m + BLOCK1 - 1) / BLOCK1);
    // printf("liv_dimgrid x: %d\n", m / BLOCK1 + ((m % BLOCK1) == 0 ? 0 : 1));
    scaleQ<<<dimGridScaleQ, dimBlockScaleQ>>>(QGPU, RGPU, m, n, k, S);

    dimGrid = dim3((n - k + blocknorm) / blocknorm, 1, 1);
    dimBlock = dim3(blocknorm, 1, 1);
    /*printf("\nkernel scaleR\n");
    printf("dimgrid x: %d\n", (n - k + BLOCK1) / BLOCK1);
    printf("liv_dimgrid x: %d\n", (n - k) / BLOCK1 + (((n - k) % BLOCK1) == 0 ? 0 : 1));*/
    scaleR<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k, S);

    // Terzo step secondo il paper
    /*printf("dimgrid y: %d\n", (n - k + BLOCK2Y) / BLOCK2Y);
    printf("liv_dimgrid y: %d\n", (n - k) / BLOCK2Y + (((n - k) % BLOCK2Y) == 0 ? 0 : 1));
    dimGrid = dim3(1, (n - k + BLOCK2Y) / BLOCK2Y, 1);
    dimBlock = dim3(BLOCK2X, BLOCK2Y, 1);
    update<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);*/

    // Terzo step secondo me
    dimGrid = dim3((n - k) / blockx + (((n - k) % blockx) == 0 ? 0 : 1), m / blocky + ((m % blocky) == 0 ? 0 : 1), 1);
    dimBlock = dim3(blockx, blocky, 1);
    step3<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);
  }

  cudaMemcpy(Q, QGPU, m * n * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(R, RGPU, n * n * sizeof(double), cudaMemcpyDeviceToHost);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Tempi esecuzione sul divice: %8.2f ms (%3.3fs)\n", time, time / 1e3);

#ifdef TIMING
  fprintf(f, "%fms  %fs (%d %d) %d\n", time, time / 1e3, blockx, blocky, blocknorm); // save result in file
#endif

  cudaFree(QGPU);
  cudaFree(RGPU);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

#ifdef TIMING
  fclose(f);
#endif

  return 0;
}