#include "BvhNode.cuh"
#include "HitRecord.cuh"
#include "Ray.cuh"
BvhNode::BvhNode(HitableList* list, curandState* state)
{
  int axis = static_cast<int>(curand_uniform(state)) * 3;
  axis = glm::clamp(axis, 0, 2);

  if (list->mListSize == 0)
  {
    mLeft = nullptr;
    mRight = nullptr;

  }
  else if (list->mListSize == 1)
  {
    mLeft = list->mList[0];
    mRight = list->mList[0];
  }
}

__device__
bool BvhNode::hit(const Ray& r, float tMin, float tMax, HitRecord& record) {
  return true;
}