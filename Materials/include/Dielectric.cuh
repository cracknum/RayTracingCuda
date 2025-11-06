#ifndef DIELECTRIC_CUH
#define DIELECTRIC_CUH
#include "Material.cuh"
#include "MaterialExports.hpp"

class MATERIAL_API Dielectric: public Material {
public:
  __device__
  Dielectric(float refractIndex);
  __device__
  ~Dielectric() override = default;
  __device__
  bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color,
    Ray& scatterRay) override;

private:
  float mRefractIndex;
};



#endif //DIELECTRIC_CUH
