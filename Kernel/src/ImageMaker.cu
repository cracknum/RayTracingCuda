#include "ImageMaker.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>

namespace MODULE_NAME
{
    namespace kernelCode
    {
        __global__ void makeImage(int width, int height, unsigned char *imageData)
        {
            int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
            int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

            if (xIndex >= width || yIndex >= height)
            {
                return;
            }

            float x = static_cast<float>(xIndex) / width;
            float y = static_cast<float>(yIndex) / height;

            int pixelIndex = (width * yIndex + xIndex) * 3;

            unsigned char red = static_cast<unsigned char>(x * 255);
            unsigned char blue = static_cast<unsigned char>(y * 255);
            unsigned char green = static_cast<unsigned char>(x * y * 255);

            imageData[pixelIndex] = red;
            imageData[pixelIndex + 1] = blue;
            imageData[pixelIndex + 2] = green;
        }

    }
    ImageMaker::ImageMaker()
    :hostId(nullptr)
    {
    }

    ImageMaker::~ImageMaker()
    {
        if (hostId)
        {
            cudaFreeHost(hostId);
            hostId = nullptr;
        }
        
    }

    void ImageMaker::makeImage(int width, int height)
    {
        unsigned char *devId;

        cudaMallocHost(&hostId, width * height * 3);
        memset(hostId, 0, sizeof(unsigned char) * width * height * 3);
        cudaMalloc(&devId, width * height * 3);
        cudaMemset(devId, 0, sizeof(unsigned char) * width * height * 3);

        dim3 threadBlock(8, 8, 1);
        dim3 gridBlock((width + 7) / 8, (height + 7) / 8, 1);
        kernelCode::makeImage<<<gridBlock, threadBlock>>>(width, height, devId);

        cudaDeviceSynchronize();
        cudaMemcpy(hostId, devId, sizeof(unsigned char) * width * height * 3, cudaMemcpyDeviceToHost);

        cudaFree(devId);
    }
}
