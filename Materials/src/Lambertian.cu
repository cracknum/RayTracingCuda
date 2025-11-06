#include "Lambertian.cuh"

#include "HitRecord.cuh"
#include "Ray.cuh"
Lambertian::Lambertian(const Color& albedo)
{
  mAlbedo = albedo;
}

__device__
bool Lambertian::scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
{
  glm::vec3 scatterDirection = ramdomHemisphere(randState, record.normal);
  if (isNearZero(scatterDirection))
  {
    scatterDirection = record.normal;
  }
  scatterRay = Ray(record.point, scatterDirection, ray.renderTime());
  color = mAlbedo;
  return true;
}