#ifndef FUZZYMETALREFLECTION_H
#define FUZZYMETALREFLECTION_H
#include "Material.cuh"

class FuzzyMetalReflection final: public Material {
public:
  __device__
  FuzzyMetalReflection(const Color& color, float fuzzy);
  __device__
  ~FuzzyMetalReflection() override = default;

  __device__
  bool scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color,
      Ray& scatterRay) override;

private:
  float mFuzzy;
};



#endif //FUZZYMETALREFLECTION_H
