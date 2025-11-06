#ifndef SPHERE_CUDA_H
#define SPHERE_CUDA_H
#include "Hitable.cuh"
#include <glm/glm.hpp>
#include "Ray.cuh"

class Material;
class Sphere: public Hitable
{
    public:
    __device__ Sphere();
    __device__ ~Sphere();
    __device__ Sphere(const glm::vec3& center, float radius, Material* material);
    __device__ Sphere(const glm::vec3& startCenter, const glm::vec3& endCenter, float radius, Material* material);
    __device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const override;

    Ray mCenter;
    float mRadius;
    Material* mMaterial;
    
};
#endif