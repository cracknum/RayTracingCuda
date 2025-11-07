#ifndef INTERVAL_CUH
#define INTERVAL_CUH
#include <cuda_runtime.h>

enum class IntervalType : unsigned int
{
  EMPTY,
  UNIVERSE
};

template <typename Type>
class Interval
{
  public:
  __device__ __forceinline__ Interval(const Type& min, const Type& max, const IntervalType& type = IntervalType::UNIVERSE)
    : mMin(min)
    , mMax(max)
    , mType(type)
  {
  }
  __device__ __forceinline__ Interval(const Interval& x, const Interval& y)
  {
    mMin = min(x.mMin, y.mMin);
    mMax = max(x.mMax, y.mMax);
  }
  __device__ ~Interval() = default;
  __device__ __forceinline__ Type clamp(const Type& v) const { return min(max(v, mMin), mMax); }

  __device__ __forceinline__ auto expand(const Type& delta) const
  {
    Type half = delta * 0.5f;
    return Interval(mMin - half, mMax + half);
  }

  Type mMin;
  Type mMax;
  IntervalType mType;
};

#endif // INTERVAL_CUH
