#ifndef AABB_CUH
#define AABB_CUH
#include <cuda_runtime.h>

template<typename Type>
class Interval
{
  __device__ Interval(const Type& min, const Type& max)
    : mMin(min)
    , mMax(max)
  {
  }
  __device__ ~Interval() = default;
  __device__ __forceinline__ Type clamp(const Type& v) const
  {
    return min(max(v, mMin), mMax);
  }

  __device__ __forceinline__ auto expand(const Type& delta) const
  {
    Type half = delta * 0.5f;
    return Interval(mMin - half, mMax + half);
  }

  Type mMin;
  Type mMax;
};
class AABB
{
};

#endif // AABB_CUH
