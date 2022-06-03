/*
 ============================================================================
 Author        : G. Barlas, Gus Mendez & Roberto Figueroa
 To build use  : make constant
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include "common/pgm.h"

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;

//************************************************************************
// Check return function 
//************************************************************************

#define CUDA_CHECK_RETURN(value)                                    \
{                                                                   \
  cudaError_t _m_cudaStat = value;                                  \
  if (_m_cudaStat != cudaSuccess)                                   \
  {                                                                 \
    fprintf(stderr, "Error %s at line %d in file %s\n",             \
      cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);   \
    exit(1);                                                        \
  }                                                                 \
}


//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;  //(w^2 + h^2)/2, radio max equivalente a centro -> esquina
  *acc = new int[rBins * degreeBins];            //el acumulador, conteo depixeles encontrados, 90*180/degInc = 9000
  memset (*acc, 0, sizeof (int) * rBins * degreeBins); //init en ceros
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++) //por cada pixel
    for (int j = 0; j < h; j++) //...
      {
        int idx = j * w + i;
        if (pic[idx] > 0) //si pasa thresh, entonces lo marca
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++; //+1 para este radio r y este theta
                theta += radInc;
              }
          }
      }
}

//*****************************************************************
// Usar memoria constante para la tabla de senos y cosenos
// inicializarlo en main y pasarlo al device
__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];


// Kernel memoria Constante
__global__ void GPU_HoughTranShared(unsigned char *pic, int w, int h, int *acc, float rMax, float rScale)
{
  // Calculo global ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block
  
  int i;
  int locID = threadIdx.x;
  int xCent = w / 2;
  int yCent = h / 2;

  // TODO: Explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  // R// xyz
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  // Use shared memory here for acc variable
  __shared__ int localAcc[degreeBins * rBins];
  // Initialize
  for (i = locID; i < degreeBins * rBins; i += blockDim.x)
    localAcc[i] = 0;

  // warps sync
  __syncthreads ();

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;

      // TODO: Debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
      // R// xyz
      atomicAdd (localAcc + (rIdx * degreeBins + tIdx), 1);
    }
  }

  // warps sync again
  __syncthreads ();

  // atomic op, add local acc to the global memory acc
  for (i = locID ; i < degreeBins * rBins ; i += blockDim.x)
    atomicAdd (acc + i, localAcc[i]);

}


//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  PGMImage inImg (argv[1]);
  cudaEvent_t start, stop;
  float time;

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos(rad);
    pcSin[i] = sin(rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // Eventualmente volver memoria global
  cudaMemcpyToSymbol(d_Cos, pcCos, sizeof (float) * degreeBins);
  cudaMemcpyToSymbol(d_Sin, pcSin, sizeof (float) * degreeBins);

  // setup and copy data from host to device
  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels; // h_in contiene los pixeles de la imagen

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads
  //1 thread por pixel
  int blockNum = ceil (w * h / 256);
  //Get time with events
  CUDA_CHECK_RETURN( cudaEventCreate(&start) );
  CUDA_CHECK_RETURN( cudaEventCreate(&stop) );
  CUDA_CHECK_RETURN( cudaEventRecord(start, 0) );

  // NOTE: We're not passing d_Sin & d_Cos to the kernel, it's constant memory!
  GPU_HoughTranShared <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale);

  CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
  CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
  CUDA_CHECK_RETURN( cudaEventElapsedTime(&time, start, stop) );

  cudaDeviceSynchronize();

  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

  // compare CPU and GPU results
  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");
  printf("EXEC TIME:  %3.1f ms \n", time);

  // Clean-up
  cudaFree ((void *) d_Cos);
  cudaFree ((void *) d_Sin);
  cudaFree ((void *) d_in);
  cudaFree ((void *) d_hough);
  free (h_hough);
  free (cpuht);
  free (pcCos);
  free (pcSin);
  cudaDeviceReset ();

  return 0;
}
