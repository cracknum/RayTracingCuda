#ifndef SPHERE_CUDA_H
#define SPHERE_CUDA_H
#include "Hitable.cuh"
#include <glm/glm.hpp>
class Sphere: public Hitable
{
    public:
    __device__ Sphere();
    __device__ Sphere(const glm::vec3& center, float radius);
    __device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const override;

    glm::vec3 mCenter;
    float mRadius;
};
#endif