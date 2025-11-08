#ifndef SPHERE_CUDA_H
#define SPHERE_CUDA_H
#include "Hitable.cuh"
#include "ObjectExports.hpp"
#include "Ray.cuh"
#include <glm/glm.hpp>

class Material;
class OBJECT_API Sphere : public Hitable
{
public:
  __device__ Sphere();
  __device__ ~Sphere();
  __device__ Sphere(const glm::vec3& center, float radius, Material* material);
  __device__ void updateBoundingBox(const glm::vec3& center, float radius);
  __device__ Sphere(
    const glm::vec3& startCenter, const glm::vec3& endCenter, float radius, Material* material);
  __device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) override;
  __device__ glm::vec2 getSphereUV(const glm::vec3& point) const;
  Ray mCenter;
  float mRadius;
  Material* mMaterial;
};
#endif