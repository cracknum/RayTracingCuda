#ifndef METAL_H
#define METAL_H
#include "Material.cuh"
#include "MaterialExports.hpp"

class MATERIAL_API Metal final: public Material {
public:
  __device__
  Metal(const Color& albedo);
  __device__
  ~Metal() override = default;

  __device__
  bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color,
    Ray& scatterRay) override;
};



#endif //METAL_H
