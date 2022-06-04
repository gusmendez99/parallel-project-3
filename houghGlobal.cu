/*
 ============================================================================
 Author        : G. Barlas
 To build use  : make global
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


// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{
  // Calculo global ID
  int gloID = blockIdx.x * blockDim.x + threadIdx.x;
  if (gloID > w * h) return;      // in case of extra threads in block

  int xCent = w / 2;
  int yCent = h / 2;

  // TODO: Explicar bien bien esta parte. Dibujar un rectangulo a modo de imagen sirve para visualizarlo mejor
  /* R// Para calcular xCoord se debe realizar el módulo de gloID con el ancho de la imagen
   con el fin de poder ubicar en qué posición en el eje x se encuentra nuestro hilo.
   Luego, se realiza un corrimiento respecto a centro de la imagen.
   Para la coordenada en y se debe divide el id de nuestro hilo por el ancho
   esto nos ubica en el eje vertical. Luego este valor es restado respecto al centro de la image
   y con esto se obtiene las coordenadas a partir del id global de nuestro hilo
   que son relativas a nuestro plano

  */
  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;

  if (pic[gloID] > 0)
  {
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
      //float r = xCoord * cos(tIdx) + yCoord * sin(tIdx); //probar con esto para ver diferencia en tiempo
      float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
      int rIdx = (r + rMax) / rScale;
      // TODO: Debemos usar atomic, pero que race condition hay si somos un thread por pixel? explique
      // R// Se debe utilizar atomic ya que estamos modificando el arreglo global, el cual es compartido
      // por todos los hilos. De esta manera garantizamos que la transacción se realice adecuadamente y no
      // ocasione race conditions.
      atomicAdd (acc + (rIdx * degreeBins + tIdx), 1);
    }
  }
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

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;

  // TODO eventualmente volver memoria global
  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);

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

  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);

  CUDA_CHECK_RETURN( cudaEventRecord(stop, 0) );
  CUDA_CHECK_RETURN( cudaEventSynchronize(stop) );
  CUDA_CHECK_RETURN( cudaEventElapsedTime(&time, start, stop) );

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
