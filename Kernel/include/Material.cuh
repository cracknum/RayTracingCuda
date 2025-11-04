#ifndef MATERIAL_H
#define MATERIAL_H
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/glm.hpp>

class Ray;
class HitRecord;

class Material
{
public:
  using Color = glm::vec3;
  __device__
  virtual ~Material() = default;
  __device__
  virtual bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay) = 0;
  /**
   * @param in 入射方向
   * @param normal 法线，需要单位向量
   * @return 出射方向
   */
  __device__ __forceinline__
  glm::vec3 reflect(const glm::vec3& in, const glm::vec3& normal)
  {
    return in - 2 * glm::dot(in, normal) * normal;
  }

  __device__
  glm::vec3 randomUnitVector(curandState* state)
  {
    glm::vec3 unitVector = {0.0f, 0.0f, 0.0f};
    float length = FLT_MAX;
    do
    {
      unitVector.x = curand_uniform(state) * 2.0f - 1.0f;
      unitVector.y = curand_uniform(state) * 2.0f - 1.0f;
      unitVector.z = curand_uniform(state) * 2.0f - 1.0f;
      length = glm::length(unitVector);
    }while (length <= 1 && length >= 1e-160);

    return glm::normalize(unitVector);
  }

  __device__
  glm::vec3 ramdomHemisphere(curandState* state, const glm::vec3 normal)
  {
    glm::vec3 onUnitSphere = randomUnitVector(state) + normal;
    if (glm::dot(onUnitSphere, normal) > 0.0f)
    {
      return onUnitSphere;
    }

    return -onUnitSphere;
  }
  Color mAlbedo = {0.0f, 0.0f, 0.0f};
};

#endif //MATERIAL_H
