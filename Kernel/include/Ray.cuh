#ifndef RAY_CUDA_H
#define RAY_CUDA_H
#include <device_launch_parameters.h>
#include <cuda.h>
#include <glm/glm.hpp>

class Ray
{
public:
    __device__ __forceinline__
    Ray() {}

    __device__ __forceinline__
    Ray(const glm::vec3 &origin, const glm::vec3 &direction)
        : mOrigin(origin), mDirection(direction)
    {
    }

    __device__ __forceinline__ glm::vec3 origin() const
    {
        return mOrigin;
    }

    __device__ __forceinline__ glm::vec3 direction() const
    {
        return mDirection;
    }

    __device__ __forceinline__ glm::vec3 pointAtParameter(float t) const
    {
        return mOrigin + mDirection * t;
    }

private:
    glm::vec3 mOrigin;
    glm::vec3 mDirection;
};
#endif