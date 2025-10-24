#include "HitableList.cuh"

__device__ HitableList::HitableList() {}

__device__ HitableList::HitableList(Hitable** l, int n)
{
  mList = l;
  mListSize = n;
}

__device__ HitableList::~HitableList() {}

__device__ bool HitableList::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const
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