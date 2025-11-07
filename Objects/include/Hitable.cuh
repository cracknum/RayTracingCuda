#ifndef HITABLE_CUDA_H
#define HITABLE_CUDA_H
#include "AABB.cuh"
#include "ObjectExports.hpp"
#include "Ray.cuh"
#include <glm/glm.hpp>
class Material;
class HitRecord;

class OBJECT_API Hitable
{
public:
  __device__ Hitable() {};
  __device__ virtual ~Hitable(){};
  __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) = 0;

  __device__ virtual AABB bounding_box() {return mBoundingBox;};

protected:
  AABB mBoundingBox;
};

#endif // HITABLE_CUDA_H