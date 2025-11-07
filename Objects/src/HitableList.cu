#include "HitableList.cuh"
#include "HitRecord.cuh"

__device__ HitableList::HitableList() {}

__device__ HitableList::HitableList(Hitable** l, int n)
{
  mList = l;
  mListSize = n;

  for (int i = 0; i < n; i++)
  {
    mBoundingBox = AABB(mBoundingBox, l[i]->bounding_box());
  }
  
}

__device__ HitableList::~HitableList() {}

__device__ bool HitableList::hit(const Ray& r, float tMin, float tMax, HitRecord& record)
{
  HitRecord tRecord;
  bool hitAnything = false;
  float closestSoFar = tMax;

  for (int i = 0; i < mListSize; i++)
  {
    if (mList[i]->hit(r, tMin, closestSoFar, tRecord))
    {
      hitAnything = true;
      closestSoFar = tRecord.t;
      record = tRecord;
    }
  }

  return hitAnything;
}