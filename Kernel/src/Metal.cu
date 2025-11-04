#include "Metal.cuh"

#include "Hitable.cuh"
Metal::Metal(const Color& albedo)
{
  mAlbedo = albedo;
}
bool Metal::scatter(
  curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
{
  (void)randState;

  glm::vec3 scatterDirection = reflect(ray.direction(), record.normal);
  scatterRay = Ray(record.point, scatterDirection);
  color = mAlbedo;

  return true;
}