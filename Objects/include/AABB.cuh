#ifndef AABB_CUDA_H
#define AABB_CUDA_H
#include "Interval.cuh"
#include <float.h>

class Ray;

class AABB
{
public:
  __device__ __forceinline__ AABB()
    : AABB(Interval<float>(FLT_MAX, -FLT_MAX), Interval<float>(FLT_MAX, -FLT_MAX),
        Interval<float>(FLT_MAX, -FLT_MAX))
  {
  }
  __device__ __forceinline__ AABB(
    const Interval<float>& x, const Interval<float>& y, const Interval<float>& z)
    : intervals{ x, y, z }
  {
  }

  __device__ __forceinline__ AABB(const AABB& aabb0, const AABB& aabb1)
    : intervals{ Interval<float>(aabb0.intervals[0], aabb1.intervals[0]),
      Interval<float>(aabb0.intervals[1], aabb1.intervals[1]),
      Interval<float>(aabb0.intervals[2], aabb1.intervals[2]) }
  {
  }

  __device__ bool hit(const Ray& ray, Interval<float> interval);

  __device__ __forceinline__ Interval<float> axisInterval(int i) { return intervals[i]; }

  Interval<float> intervals[3];
};
#endif