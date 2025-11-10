#ifndef MATERIAL_H
#define MATERIAL_H
#include "MaterialExports.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <glm/ext/quaternion_geometric.hpp>
#include <glm/glm.hpp>

class Ray;
struct HitRecord;

class MATERIAL_API Material
{
public:
  using Color = glm::vec3;
  __device__ 
  Material() = default;
  __device__
  virtual ~Material() = default;
  __device__
  virtual bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay) = 0;
  __device__
  virtual Color emitted(float u, float v, const glm::vec3& point){return Color(0, 0, 0);}
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

  /**
   *@brief 材质的折射率计算
   *@param uv 光线的入射方向
   *@param n 击中后的位置的法向量
   *@param etai_over_etat 材质的折射率
   */
  __device__ __forceinline__
  glm::vec3 refract(const glm::vec3& uv, const glm::vec3& n, float etai_over_etat)
  {
    // 来源于snell折射公式的推导
    auto cos_theta = fminf(glm::dot(-uv, n), 1.0f);
    glm::vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    float perp_len_sq = glm::dot(r_out_perp, r_out_perp);
    if (perp_len_sq > 1.0f)
    {
      return reflect(uv, n);
    }
    
    glm::vec3 r_out_parallel = -sqrtf(1.0f - perp_len_sq) * n;

    return r_out_perp + r_out_parallel;
  }

  /**
   *@brief 对于snell's low的近似计算（Schlick Approximation）
   *@param cosine 入射角的余弦值
   *@param refractIndex 反射率
   *@return 反射率
   */
  __device__ __forceinline__ float reflectance(float cosine, float refractIndex)
  {
    auto r0 = (1 - refractIndex) / (1 + refractIndex);
    r0 = r0 * r0;

    return r0 + (1 - r0) * powf((1 - cosine), 5.0f);
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
  cudaTextureObject_t mTexture;
  Color mAlbedo = {0.0f, 0.0f, 0.0f};
};

#endif //MATERIAL_H
