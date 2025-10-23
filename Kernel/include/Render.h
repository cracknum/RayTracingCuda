#ifndef RENDER_CUDA_H
#define RENDER_CUDA_H
#include <glm/glm.hpp>
#include "KernelExports.hpp"

namespace Kernel
{
    struct KERNEL_API ImageInfo
    {
        unsigned char* mColor;
        int width;
        int height;
    };

    struct KERNEL_API SpaceImageInfo
    {
        glm::vec3 mLowerLeftCorner;
        glm::vec3 mHorizontal;
        glm::vec3 mVertical;
    };
    void KERNEL_API render(ImageInfo& imageInfo, const SpaceImageInfo& spaceImageInfo, const glm::vec3& rayOrigin);

}

#endif