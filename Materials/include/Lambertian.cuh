#ifndef LAMBERTIAN_H
#define LAMBERTIAN_H
#include "Material.cuh"
#include "MaterialExports.hpp"

class MATERIAL_API Lambertian final : public Material {
public:
  __device__
  explicit Lambertian(const Color& albedo);
  __device__
  explicit Lambertian(cudaTextureObject_t texture);
  __device__
  ~Lambertian() override = default;
  __device__
  bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay) override;

protected:
  __device__ __forceinline__
  bool isNearZero(const glm::vec3& vector)
  {
    return fabs(vector.x) < 1e-6 && fabs(vector.y) < 1e-6 && fabs(vector.z) < 1e-6;
  }
};



#endif //LAMBERTIAN_H
