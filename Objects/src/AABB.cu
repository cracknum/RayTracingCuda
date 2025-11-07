#include "AABB.cuh"
#include "Ray.cuh"

__device__ bool AABB::hit(const Ray& ray, Interval<float> interval)
{
  const glm::vec3& origin = ray.origin();
  const glm::vec3& direction = ray.direction();
  #pragma unroll(3)
  for (int i = 0; i < 3; i++)
  {
    const Interval<float>& axis = axisInterval(i); 

    auto t0 = (axis.mMin - origin[i]) / direction[i];
    auto t1 = (axis.mMax - origin[i]) / direction[i];

    if (t0 < t1)
    {
      if (t0 > interval.mMin)
      {
        interval.mMin = t0;
      }
      if (t1 < interval.mMax)
      {
        interval.mMax = t1;
      }
    }else
    {
      if (t1 > interval.mMin)
      {
        interval.mMin = t1;
      }
      if (t0 < interval.mMax)
      {
        interval.mMax = t0;
      }
    }

    if (interval.mMax < interval.mMin)
    {
      return false;
    }
  }

  return true;
  
}