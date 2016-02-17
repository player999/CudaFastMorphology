#include <npp.h>

#define BORDER_VALUE 255

#define PRINT_ON

#ifndef PRINTF
# ifndef PRINT_ON
#  define PRINTF(...) ((void)0)
# else
#  define PRINTF(fmt,...) (printf(fmt, ## __VA_ARGS__))
# endif
#endif



template <class dataType, morphOperation MOP>
__global__ void _verticalVHGWKernel(const dataType *img, int imgStep, dataType *result,
                                    int resultStep, unsigned int width, unsigned int height,
                                        unsigned int size, NppiSize borderSize) {
    const unsigned int y      = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	  const unsigned int step   = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int startx = __umul24(step,size);

    if (y >= height || startx > width)
        return;

    const dataType *lineIn = img+y;
    dataType *lineOut      = result+y;

    const unsigned int center  = startx + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center*imgStep];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[(center-k)*imgStep];
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = (center+k < height+size-1) ? lineIn[(center+k)*imgStep] : 255;
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    } else {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[__umul24(center-k,imgStep)];
            minarray[size-1-k] = max(minarray[size-k], nextMin);

            nextMin = lineIn[__umul24(center+k,imgStep)];
            minarray[size-1+k] = max(minarray[size+k-2], nextMin);
        }
    }

    int diff = height - startx;
    if (diff > 0) {
        lineOut += startx*resultStep;
        lineOut[0] = minarray[0];

        for(k=1; k < size-1; ++k) {
            if (diff > k) {
                lineOut[k*resultStep] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
            }
        }

        if (diff > size-1) {
            lineOut[(size-1)*resultStep] = minarray[2*(size-1)];
        }
    }
}

#if 0

template <class dataType, morphOperation MOP>
__global__ void _horizontalVHGWKernel(const dataType *img, int imgStep, dataType *result,
                                    int resultStep, unsigned int width, unsigned int height,
                                        unsigned int size, NppiSize borderSize) {
    const unsigned int x      = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    const unsigned int step   = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const unsigned int starty = __umul24(step,size);

    if (x >= width || starty > height)
        return;

    const dataType *lineIn = img+x*imgStep;
    dataType *lineOut      = result+x*resultStep;
    const unsigned int center  = starty + (size-1);

    dataType minarray[512];
    minarray[size-1] = lineIn[center];

    dataType nextMin;
    unsigned int k;
    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[center-k];
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = (center+k < width+size-1) ? lineIn[center+k] : BORDER_VALUE;
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    } else {
        for(k=1;k<size; ++k) {
            nextMin = lineIn[center-k];
            minarray[size-1-k] = max(minarray[size-k], nextMin);

            nextMin =lineIn[center+k];
            minarray[size-1+k] = max(minarray[size+k-2], nextMin);
        }
    }

    int diff = width - starty;
    if (diff > 0) {
        lineOut += starty;
        lineOut[0] = minarray[0];

        for(k=1; k < size-1; ++k) {
            if (diff > k) {
                lineOut[k] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
            }
        }

        if (diff > size-1) {
            lineOut[size-1] = minarray[2*(size-1)];
        }
    }
}

#else
# if 1
template <class dataType, morphOperation MOP>
__global__ void _horizontalVHGWKernel(const dataType *img, int imgStep,
                                      dataType *result, int resultStep,
                                      unsigned int width, unsigned int height,
                                      unsigned int size, NppiSize borderSize)
{
    #define LINEC 13
    #define LINES 1040
    __shared__ dataType imHx[LINEC * LINES];
    __shared__ dataType imGx[LINEC * LINES];
    dataType *imHxPtr, *imGxPtr;
    dataType *imHxStepPtr, *imGxStepPtr;
    uint32_t ptroffset;

    dataType localSrc[13];
    dataType localGx[13], localHx[13];
    uint32_t startx = __umul24(size, threadIdx.x);
    uint32_t imline = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    const dataType *srcptr;
    char pred = !(imline >= height) && !((startx - size) >= width);
    const uint32_t *ptr32;
    uint32_t src32[4];
    uint8_t *ptr8;
    dataType *dstptr;

    //Load data from global memory to shared memory
    ptroffset = threadIdx.y * LINES;
    imGxPtr = imGx + ptroffset;
    imHxPtr = imHx + ptroffset;
    srcptr = img + imline * imgStep + startx;
    imGxStepPtr = imGxPtr + startx;
    imHxStepPtr = imHxPtr + startx;
    ptr32 = (const uint32_t *) ((uint32_t)srcptr & (0xFFFFFFFC));
    ptr8 = (uint8_t *) src32;

    if (pred) {
      asm("prefetch.global.L1 [%0];"::"r"(srcptr));
      src32[0] = __ldg(ptr32 + 0);
      src32[1] = __ldg(ptr32 + 1);
      src32[2] = __ldg(ptr32 + 2);
      src32[3] = __ldg(ptr32 + 3);

      ptr8 += (srcptr-(dataType *)ptr32);

      localSrc[0] = ptr8[0];
      localSrc[1] = ptr8[1];
      localSrc[2] = ptr8[2];
      localSrc[3] = ptr8[3];
      localSrc[4] = ptr8[4];
      localSrc[5] = ptr8[5];
      localSrc[6] = ptr8[6];
      localSrc[7] = ptr8[7];
      localSrc[8] = ptr8[8];
      localSrc[9] = ptr8[9];
      localSrc[10] = ptr8[10];
      localSrc[11] = ptr8[11];
      localSrc[12] = ptr8[12];

      //Processing
      imGxStepPtr[0] = localGx[0] = localSrc[0];
      imGxStepPtr[1] = localGx[1] = max(localGx[0], localSrc[1]);
      imGxStepPtr[2] = localGx[2] = max(localGx[1], localSrc[2]);
      imGxStepPtr[3] = localGx[3] = max(localGx[2], localSrc[3]);
      imGxStepPtr[4] = localGx[4] = max(localGx[3], localSrc[4]);
      imGxStepPtr[5] = localGx[5] = max(localGx[4], localSrc[5]);
      imGxStepPtr[6] = localGx[6] = max(localGx[5], localSrc[6]);
      imGxStepPtr[7] = localGx[7] = max(localGx[6], localSrc[7]);
      imGxStepPtr[8] = localGx[8] = max(localGx[7], localSrc[8]);
      imGxStepPtr[9] = localGx[9] = max(localGx[8], localSrc[9]);
      imGxStepPtr[10] = localGx[10] = max(localGx[9], localSrc[10]);
      imGxStepPtr[11] = localGx[11] = max(localGx[10], localSrc[11]);
      imGxStepPtr[12] = localGx[12] = max(localGx[11], localSrc[12]);

      imHxStepPtr[12] = localHx[12] = localSrc[12];
      imHxStepPtr[11] = localHx[11] = max(localHx[12], localSrc[11]);
      imHxStepPtr[10] = localHx[10] = max(localHx[11], localSrc[10]);
      imHxStepPtr[9] = localHx[9] = max(localHx[10], localSrc[9]);
      imHxStepPtr[8] = localHx[8] = max(localHx[9], localSrc[8]);
      imHxStepPtr[7] = localHx[7] = max(localHx[8], localSrc[7]);
      imHxStepPtr[6] = localHx[6] = max(localHx[7], localSrc[6]);
      imHxStepPtr[5] = localHx[5] = max(localHx[6], localSrc[5]);
      imHxStepPtr[4] = localHx[4] = max(localHx[5], localSrc[4]);
      imHxStepPtr[3] = localHx[3] = max(localHx[4], localSrc[3]);
      imHxStepPtr[2] = localHx[2] = max(localHx[3], localSrc[2]);
      imHxStepPtr[1] = localHx[1] = max(localHx[2], localSrc[1]);
      imHxStepPtr[0] = localHx[0] = max(localHx[1], localSrc[0]);
    }

    __syncthreads();
    if(pred) {
      //Save data fromshared memory to global memory
      dstptr = result + imline * resultStep + startx;
      dstptr[0] = max(localGx[6], imHxStepPtr[0]);
      dstptr[1] = max(localGx[7], imHxStepPtr[1]);
      dstptr[2] = max(localGx[8], imHxStepPtr[2]);
      dstptr[3] = max(localGx[9], imHxStepPtr[3]);
      dstptr[4] = max(localGx[10], imHxStepPtr[4]);
      dstptr[5] = max(localGx[11], imHxStepPtr[5]);
      dstptr[6] = max(localGx[12], localHx[0]);
      dstptr[7] = max(imGxStepPtr[7], localHx[1]);
      dstptr[8] = max(imGxStepPtr[8], localHx[2]);
      dstptr[9] = max(imGxStepPtr[9], localHx[3]);
      dstptr[10] = max(imGxStepPtr[10], localHx[4]);
      dstptr[11] = max(imGxStepPtr[11], localHx[5]);
      dstptr[12] = max(imGxStepPtr[12], localHx[6]);
    }
}
# else
template <class dataType, morphOperation MOP>
__global__ void _horizontalVHGWKernel(const  dataType *img, int imgStep,
                                      dataType *result, int resultStep,
                                      unsigned int width, unsigned int height,
                                      unsigned int size, NppiSize borderSize)
{
    #define LINEC 13
    #define LINES 1040
    __shared__ dataType imHx[LINEC * LINES];
    __shared__ dataType imGx[LINEC * LINES];
    dataType *imHxPtr, *imGxPtr;
    dataType *imHxStepPtr, *imGxStepPtr;
    uint32_t ptroffset;
    uint32_t j;

    dataType localSrc[13];
    uint32_t startx = __umul24(size, threadIdx.x);
    uint32_t imline = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
    dataType *dstptr;
    const __restrict__ dataType *srcptr;
    char pred = !(imline >= height) && !((startx - size) >= width);

    //Load data from global memory to shared memory
    ptroffset = threadIdx.y * LINES;
    imGxPtr = imGx + ptroffset;
    imHxPtr = imHx + ptroffset;
    srcptr = img + imline * imgStep + startx;
    imGxStepPtr = imGxPtr + startx;
    imHxStepPtr = imHxPtr + startx;

    if (pred) {
      asm("prefetch.global.L1 [%0];"::"r"(srcptr));
      for (int i = 0; i < size; i++) localSrc[i] = srcptr[i];
      //Processing
      dataType gxMax, hxMax;

      imGxStepPtr[0] = gxMax = localSrc[0];
      for (int i = 1; i < size; i++) imGxStepPtr[i] = gxMax = max(gxMax, localSrc[i]);

      imHxStepPtr[12] = hxMax = localSrc[12];
      for (int i = 11; i >= 0; i--) imHxStepPtr[i] = hxMax = max(hxMax, localSrc[i]);
    }

    __syncthreads();
    if(pred) {
      //Save data fromshared memory to global memory
      imHxStepPtr -= 6;
      imGxStepPtr += 6;
      dstptr = result + imline * resultStep + startx;
      j = 12;
      do {*(dstptr++) = max(*(imGxStepPtr++), *(imHxStepPtr++)); } while(j--);
    }
}

# endif
#endif

/*{
    dataType minarray[512];
    dataType *inputRow, *lineOut;

	const unsigned int y    = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;
	const unsigned int step = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;

    const unsigned int startx = __umul24(step,size);
    if (y >= height + size/2 || startx > width)
        return;

    inputRow = (dataType*)img + y*imgStep;
    lineOut = result + y*resultStep;

    const unsigned int windowCenter  = step*size+(size-1);
    unsigned int k;

    minarray[size-1] = inputRow[windowCenter];
    dataType nextMin;

    if (MOP == ERODE) {
        for(k=1;k<size; ++k) {
            nextMin = inputRow[windowCenter-k];
            minarray[size-1-k] = min(minarray[size-k], nextMin);

            nextMin = inputRow[windowCenter+k];
            minarray[size-1+k] = min(minarray[size+k-2], nextMin);
        }
    } else {
        for(k=1;k<size; ++k) {
            nextMin = inputRow[windowCenter-k];
            minarray[size-1-k] = max(minarray[size-k], nextMin);

            nextMin = inputRow[windowCenter+k];
            minarray[size-1+k] = max(minarray[size+k-2], nextMin);
        }
    }

    int hdiff = height - startx;
    if (0 < hdiff) {
        lineOut += startx;

        lineOut[0] = minarray[0];

        for(k=1; k < size-1; ++k) {
            if (k <= hdiff) {
                lineOut[k] = minMax<dataType, MOP>(minarray[k], minarray[k+size-1]);
            }
        }

        if (size-1 <= hdiff) {
            lineOut[size-1] = minarray[__umul24(2,size-1)];
        }
    }
}*/


template <class dataType, morphOperation MOP, vhgwDirection DIRECTION>
NppStatus _globalVHGW(const dataType * img, Npp32s imgStep, dataType * result,
                        Npp32s resultStep, NppiSize oSizeROI, unsigned int size,
                            NppiSize borderSize) {
    const unsigned int width = oSizeROI.width;
    const unsigned int height = oSizeROI.height;

    PRINTF("width %d, height %d\n", width, height);
    PRINTF("Border (w: %d , h: %d)\n", borderSize.width, borderSize.height);

    unsigned int steps;
    if (DIRECTION == VERTICAL) {
        steps = (width+size-1)/size;
        dim3 gridSize((steps+128-1)/128, (height+2-1)/2);
        dim3 blockSize(128,2);

        _verticalVHGWKernel<dataType, MOP><<<gridSize,blockSize>>>
            (img, imgStep,result, resultStep, width, height, size, borderSize);
    }
    else { // HORIZONTAL
        int linesblock;
        int lines;
        dim3 gridSize;
        dim3 blockSize;

        steps = width / size;

        lines = 16384 / width;
        if (lines * steps > 1024) lines = 1024 / steps;

        linesblock = (height % lines) ? height / lines + 1 : height / lines;

        blockSize = dim3(steps, lines);
        gridSize = dim3(1, linesblock);
        printf("Block size (%d,%d)\n", steps, lines);
        printf("Grid size (%d,%d)\n", 1, linesblock);
#if 1

        for (int i = 0; i < 100; i++) {
          _horizontalVHGWKernel<dataType, MOP><<<gridSize,blockSize>>>
            (img, imgStep,result, resultStep, width, height, size, borderSize);
          cudaDeviceSynchronize();
        }

#else
        CUmodule module;
        CUfunction function;
        CUresult err;

        const char* module_file = "horizontal13.ptx";
        const char* kernel_name = "vhgw_horizontal13";
        const char *errstr;

        printf("Loading ptx!\n");
        err = cuModuleLoad(&module, module_file);
        if (CUDA_SUCCESS != err) {
          printf("Failed to load module\n");
          cuGetErrorString(err, &errstr);
          printf("%s\n", errstr);
          exit(255);
        }

        printf("Loading function!\n");
        err = cuModuleGetFunction(&function, module, kernel_name);
        if (CUDA_SUCCESS != err) {
          printf("Failed to load function\n");
          cuGetErrorString(err, &errstr);
          printf("%s\n", errstr);
          exit(255);
        }

        printf("Launching kernel!\n");

        /*
           CUresult cuLaunchKernel (
           CUfunction f,
           unsigned int  gridDimX, unsigned int  gridDimY, unsigned int  gridDimZ,
           unsigned int  blockDimX, unsigned int  blockDimY, unsigned int  blockDimZ,
           unsigned int  sharedMemBytes,
           CUstream hStream,
           void** kernelParams,
           void** extra )
        */

        cuParamSetSize(function, 7 * 4);
        cuParamSetv(function, 0, (void *)&img, 4);
        cuParamSetv(function, 4, (void *)&imgStep, 4);
        cuParamSetv(function, 8, (void *)&result, 4);
        cuParamSetv(function, 12, (void *)&resultStep, 4);
        cuParamSetv(function, 16, (void *)&width, 4);
        cuParamSetv(function, 20, (void *)&height, 4);
        cuParamSetv(function, 24, (void *)&size, 4);
        cuFuncSetBlockShape (function, steps, lines, 1);
        cuFuncSetSharedSize (function, 33280);

        for(int i = 0; i < 100; i++) {
          cuLaunchGrid (function, 1, linesblock);

          if (CUDA_SUCCESS != err) {
            printf("Failed to launch function\n");
            cuGetErrorString(err, &errstr);
            printf("%d: %s\n", err, errstr);
            exit(255);
          }
          cudaDeviceSynchronize();
        }
#endif
    }

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
       //print the CUDA error message and exit
       PRINTF("CUDA error: %s\n", cudaGetErrorString(error));
       exit(-1);
    }

    return NPP_SUCCESS;
}

/*
    Function for writing images in .PGM format. Useful for debugging, to track image changes step by step.

    void writeImageToPGM(const char* filename, const unsigned char* dev, int devStep, unsigned int width, unsigned int height) {
    int r,c;
    unsigned char *host = (unsigned char*)malloc(width*height);

    cudaMemcpy2D((void*)host, width, dev, devStep, width, height, cudaMemcpyDeviceToHost);

    // check for error
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA writeImageToPGM error: %s\n", cudaGetErrorString(error));
       // exit(-1);
    } else {
        FILE *file;
        file = fopen(filename, "w");
        fprintf(file,"P5\n%d %d\n255\n", height, width);
        for(c = 0; c < width; c++) {
        	for(r = 0; r < height; r++) {
	            fputc(host[r*width + c],file);
	        }
        }
        fclose(file);
    }
}*/
