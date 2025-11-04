#include "FuzzyMetalReflection.cuh"

#include "Hitable.cuh"
#include "Ray.cuh"
FuzzyMetalReflection::FuzzyMetalReflection(const Color& color, float fuzzy)
{
  mAlbedo = color;
  mFuzzy = glm::clamp(fuzzy, 0.0f, 1.0f);
}

__device__
bool FuzzyMetalReflection::scatter(
  curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
{
  glm::vec3 scatterDirection = reflect(ray.direction(), record.normal);
  scatterDirection += randomUnitVector(randState) * mFuzzy;
  scatterRay = Ray(record.point, scatterDirection);
  color = mAlbedo;

  return glm::dot(scatterDirection, record.normal) > 0.0f;
}