#ifndef HITABLE_CUDA_H
#define HITABLE_CUDA_H
#include "Ray.cuh"
#include <glm/glm.hpp>

struct HitRecord
{
    float t;
    glm::vec3 point;
    glm::vec3 normal;     
};

class Hitable
{
    public:
        __device__ Hitable() {};
        __device__ virtual ~Hitable() {};
        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const = 0;
};

#endif // HITABLE_CUDA_H