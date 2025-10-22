#ifndef IMAGE_CUDA_H
#define IMAGE_CUDA_H
#include "Macro.h"
#include <string>
#include "KernelExports.hpp"
namespace MODULE_NAME
{
    class KERNEL_API ImageMaker
    {
    public:
        ImageMaker();
        ~ImageMaker();

        void makeImage(int width, int height);

        unsigned char *hostId;
    };

}
#endif