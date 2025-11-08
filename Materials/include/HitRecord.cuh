#ifndef HIT_RECORD_CUDA_CXX
#define HIT_RECORD_CUDA_CXX
#include "Material.cuh"
#include "Ray.cuh"
#include <device_launch_parameters.h>
#include <glm/glm.hpp>

struct MATERIAL_API HitRecord
{
  float t;
  glm::vec3 point;
  glm::vec3 normal;
  Material* material;
  // 是否是surface外侧
  bool front_face;
  // texture coord
  float u;
  float v;

  __device__ __forceinline__ void setFaceNormal(const Ray& r, const glm::vec3& outwardNormal)
  {
    front_face = (glm::dot(r.direction(), outwardNormal) < 0);
    normal = front_face ? outwardNormal : -outwardNormal;
  }
};
#endif