#ifndef BVHNODE_CUH
#define BVHNODE_CUH
#include "Hitable.cuh"
#include "HitableList.cuh"
#include <curand_kernel.h>

class Ray;
class HitRecord;
class BvhNode final : public Hitable
{

public:
  __device__ explicit BvhNode(HitableList* list, curandState* state);
  __device__
  bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) override;

private:
  Hitable* mLeft;
  Hitable* mRight;
};
#endif //BVHNODE_CUH
