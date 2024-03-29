#include <cuda.h>
#include "myQR.h"

//#define DEBUG
//#define TIMING
#define TIMINGPERBLOCK

#define CUDA_SAFE_CALL(call)                                        \
  {                                                                 \
    cudaError err = call;                                           \
    if (cudaSuccess != err)                                         \
    {                                                               \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString(err));         \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
  }

__global__ void mult(double *Q, double *R, int m, int n, int k);
__global__ void scaleR(double *Q, double *R, int m, int n, int k, double S);
__global__ void scaleQ(double *Q, double *R, int m, int n, int k, double S);
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
    if ((blockx % 2) != 0)
    {
      printf("NumbThreadsBlockX deve essere multiplo di 2.\n");
      return -1;
    }
    /*if (blockx < blocky)
    {
      printf("NumbThreadsBlockX deve essere maggiore di NumbThreadsBlockY\n");
      return -1;
    }*/
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
  if (i < k || i >= n)
    return;

  for (int j = tid1; j < m; j += blockDim.x)
    S += Q[IDX(j, k, m)] * Q[IDX(j, i, m)];

  // thread writes result in shared array RS
  int index = tid2 * blockDim.x + tid1;
  rs[index] = S;

  int NT = blockDim.x;

  while (NT > 1)
  {
    // first half of threads sums up
    __syncthreads();
    NT = NT >> 1;
    if (tid1 < NT)
      rs[index] += rs[index + NT];
  }

  // now thread 0 writes the result

  if (tid1 == 0)
    R[IDX(k, i, n)] = rs[index];
}

__global__ void scaleR(double *Q, double *R, int m, int n, int k, double S)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x + k;
  if (i >= k && i < n)
    R[IDX(k, i, n)] *= S;
}

__global__ void scaleQ(double *Q, double *R, int m, int n, int k, double S)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x; //+1
  if (i < m)
    Q[IDX(i, k, m)] *= S;
}

__global__ void step3(double *A, double *R, int M, int N, int k)
{
  int i = threadIdx.y + blockDim.y * blockIdx.y;         // row index
  int j = threadIdx.x + blockDim.x * blockIdx.x + k + 1; // col index + k + 1
  if (i < M && j < N)
    A[IDX(i, j, M)] -= A[IDX(i, k, M)] * R[IDX(k, j, N)];
}

int QR(double *Q, double *R, unsigned int m, unsigned int n, int blockx, int blocky, int blocknorm)
{
  double *QGPU; // Q on GPU
  double *RGPU; // R on GPU
  float time;
  dim3 dimGrid, dimGridScaleQ;
  dim3 dimBlock, dimBlockScaleQ;
  int shMem = blockx * blocky * sizeof(double);
  // Constant dimGrid and dimBlock for kernel ScaleQ
  dimGridScaleQ = dim3((m + blocknorm - 1) / blocknorm, 1, 1);
  dimBlockScaleQ = dim3(blocknorm, 1, 1);
  cudaEvent_t start, stop;

#if defined(TIMING) || defined(TIMINGPERBLOCK)
  FILE *f; // file results
  char filename[256];
  sprintf(filename, "list.gpu.%d_%d", m, n); // create file results
  f = fopen(filename, "a");
#endif

CUDA_SAFE_CALL(cudaEventCreate(&start));
CUDA_SAFE_CALL(cudaEventCreate(&stop));

#ifdef TIMINGPERBLOCK
  float timeM, timeR, timeQ, timeStep3;
  float timeMM=0.f, timeRR=0.f, timeQQ=0.f, timeSS=0.f;
  cudaEvent_t startM, stopM, startQ, stopQ, startR, stopR, startStep3, stopStep3;

  CUDA_SAFE_CALL(cudaEventCreate(&startM));
  CUDA_SAFE_CALL(cudaEventCreate(&stopM));

  CUDA_SAFE_CALL(cudaEventCreate(&startQ));
  CUDA_SAFE_CALL(cudaEventCreate(&stopQ));

  CUDA_SAFE_CALL(cudaEventCreate(&startR));
  CUDA_SAFE_CALL(cudaEventCreate(&stopR));

  CUDA_SAFE_CALL(cudaEventCreate(&startStep3));
  CUDA_SAFE_CALL(cudaEventCreate(&stopStep3));
#endif

  // Allocate on GPU
  CUDA_SAFE_CALL(cudaEventRecord(start, 0));
  CUDA_SAFE_CALL(cudaMalloc((void **)&QGPU, m * n * sizeof(double)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&RGPU, n * n * sizeof(double)));

  CUDA_SAFE_CALL(cudaMemcpy(QGPU, Q, m * n * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(RGPU, R, n * n * sizeof(double), cudaMemcpyHostToDevice));

  for (unsigned int k = 0; k < n; k++)
  {
    dimGrid = dim3((n - k + blocky) / blocky, 1, 1);
    dimGrid = dim3((n - k) / blocky + (((n - k) % blocky) == 0 ? 0 : 1));
    dimBlock = dim3(blockx, blocky, 1);

    #ifdef TIMINGPERBLOCK
      CUDA_SAFE_CALL(cudaEventRecord(startM, 0));
    #endif
    mult<<<dimGrid, dimBlock, shMem>>>(QGPU, RGPU, m, n, k);
    #ifdef TIMINGPERBLOCK
      cudaEventRecord(stopM, 0);
      cudaEventSynchronize(stopM);
      cudaEventElapsedTime(&timeM, startM, stopM);
      timeMM += timeM;
    #endif
    // fattore di scala
    double S;
    CUDA_SAFE_CALL(cudaMemcpy(&S, &RGPU[IDX(k, k, n)], sizeof(double), cudaMemcpyDeviceToHost));
    S = sqrt(S);
    S = 1.0 / S;

    #ifdef TIMINGPERBLOCK
      CUDA_SAFE_CALL(cudaEventRecord(startQ, 0));
    #endif
    scaleQ<<<dimGridScaleQ, dimBlockScaleQ>>>(QGPU, RGPU, m, n, k, S);
    #ifdef TIMINGPERBLOCK
      cudaEventRecord(stopQ, 0);
      cudaEventSynchronize(stopQ);
      cudaEventElapsedTime(&timeQ, startQ, stopQ);
      timeQQ += timeQ;
    #endif

    dimGrid = dim3((n - k + blocknorm) / blocknorm, 1, 1);
    dimBlock = dim3(blocknorm, 1, 1);

    #ifdef TIMINGPERBLOCK
      CUDA_SAFE_CALL(cudaEventRecord(startR, 0));
    #endif
    scaleR<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k, S);
    #ifdef TIMINGPERBLOCK
      cudaEventRecord(stopR, 0);
      cudaEventSynchronize(stopR);
      cudaEventElapsedTime(&timeR, startR, stopR);
      timeRR += timeR;
    #endif
    dimGrid = dim3((n - k) / blockx + (((n - k) % blockx) == 0 ? 0 : 1), m / blocky + ((m % blocky) == 0 ? 0 : 1), 1);
    dimBlock = dim3(blockx, blocky, 1);

    #ifdef TIMINGPERBLOCK
      CUDA_SAFE_CALL(cudaEventRecord(startStep3, 0));
    #endif
    step3<<<dimGrid, dimBlock>>>(QGPU, RGPU, m, n, k);
    #ifdef TIMINGPERBLOCK
      cudaEventRecord(stopStep3, 0);
      cudaEventSynchronize(stopStep3);
      cudaEventElapsedTime(&timeStep3, startStep3, stopStep3);
      timeSS += timeStep3;
    #endif
  }

  CUDA_SAFE_CALL(cudaMemcpy(Q, QGPU, m * n * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(R, RGPU, n * n * sizeof(double), cudaMemcpyDeviceToHost));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  printf("Tempi esecuzione sul divice: %8.2f ms (%3.3fs)\n", time, time / 1e3);

#ifdef TIMING
  fprintf(f, "%fms  %fs (%d %d) %d\n", time, time / 1e3, blockx, blocky, blocknorm); // save result in file
#endif

#ifdef TIMINGPERBLOCK
  fprintf(f, "M:%fms  Q:%fms  R:%fms  St3:%fms\t(%d %d) %d\n", timeMM, timeQQ, timeRR, timeSS, blockx, blocky, blocknorm);
  printf("M:%fms  Q:%fms  R:%fms  St3:%fms\t(%d %d) %d\n", timeMM, timeQQ, timeRR, timeSS, blockx, blocky, blocknorm);
#endif

  CUDA_SAFE_CALL(cudaFree(QGPU));
  CUDA_SAFE_CALL(cudaFree(RGPU));

  CUDA_SAFE_CALL(cudaEventDestroy(start));
  CUDA_SAFE_CALL(cudaEventDestroy(stop));

#ifdef TIMING
  fclose(f);
#endif

  return 0;
}