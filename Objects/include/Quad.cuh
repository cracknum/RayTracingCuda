#ifndef QUAD_CUDA_H
#define QUAD_CUDA_H
#include "Hitable.cuh"
#include "Material.cuh"
#include <glm/glm.hpp>

struct HitRecord;
class Ray;
class Material;
class Quad : public Hitable
{
public:
  __device__ Quad(const glm::vec3& Q, const glm::vec3& u, const glm::vec3& v, Material* material);
  __device__ bool hit(const Ray& r, float tMin, float tMax, HitRecord& record) override;

  protected:
  private:
  glm::vec3 mQ;
  glm::vec3 mU;
  glm::vec3 mV;
  glm::vec3 mW;
  glm::vec3 mNormal;
  float mD;
  Material* mMaterial;
};
#endif // QUAD_CUDA_H