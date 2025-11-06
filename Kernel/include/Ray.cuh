#ifndef RAY_CUDA_H
#define RAY_CUDA_H
#include <cuda.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

class Ray
{
public:
  __device__ __forceinline__ Ray() {}

  __device__ __forceinline__ Ray(
    const glm::vec3& origin, const glm::vec3& direction, float renderTime)
    : mOrigin(origin)
    , mDirection(direction)
    , mRenderTime(renderTime)
  {
  }

  __device__ __forceinline__ Ray(const glm::vec3& origin, const glm::vec3& direction)
    : Ray(origin, direction, 0)
  {
  }

  __device__ __forceinline__ glm::vec3 origin() const { return mOrigin; }

  __device__ __forceinline__ glm::vec3 direction() const { return mDirection; }

  __device__ __forceinline__ glm::vec3 pointAtParameter(float t) const
  {
    return mOrigin + mDirection * t;
  }

  __device__ __forceinline__ float renderTime() const { return mRenderTime; }

private:
  glm::vec3 mOrigin;
  glm::vec3 mDirection;
  float mRenderTime;
};
#endif