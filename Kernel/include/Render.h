#ifndef RENDER_CUDA_H
#define RENDER_CUDA_H
#include <glm/glm.hpp>
#include <device_launch_parameters.h>
#include <cuda.h>

namespace Render
{
    class Ray;
    __device__ glm::vec3 color(const Ray &r);

}

#endif