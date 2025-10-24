#ifndef HITABLE_LIST_CUDA_H
#define HITABLE_LIST_CUDA_H
#include "Hitable.cuh"
class HitableList: public Hitable
{
    public:
    __device__ HitableList();
    __device__ HitableList(Hitable **l, int n);
    __device__ ~HitableList();
    __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const override;

    Hitable** mList;
    int mListSize;
};
#endif