#ifndef HITABLE_CUDA_H
#define HITABLE_CUDA_H
#include "Ray.cuh"
#include <glm/glm.hpp>

class Material;

struct HitRecord
{
    float t;
    glm::vec3 point;
    glm::vec3 normal;     
    Material* material;
  // 是否是surface外侧
    bool front_face;

  __device__  __forceinline__ void   setFaceNormal(const Ray& r, const glm::vec3& outwardNormal)
  {
    front_face = (glm::dot(r.direction(), outwardNormal) < 0);
    normal = front_face ? outwardNormal : -outwardNormal;
  }
};

class Hitable
{
    public:
        __device__ Hitable() {};
        __device__ virtual ~Hitable() {};
        __device__ virtual bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) const = 0;

};

#endif // HITABLE_CUDA_H