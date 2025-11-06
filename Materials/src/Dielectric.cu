#include "Dielectric.cuh"

#include "HitRecord.cuh"
__device__ Dielectric::Dielectric(float refractIndex)
  :mRefractIndex(refractIndex)
{}

__device__
bool Dielectric::scatter(
  curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
{
  color = Color(1.0f, 1.0f, 1.0f);
  // 材质是在内部还是在外部，如果在内部则为本身的折射率，如果在外部则会被内部材质的折射率除去
  float ri = record.front_face ? (1.0f / mRefractIndex) : mRefractIndex;

  glm::vec3 unitDirection = glm::normalize(ray.direction());

  float cos_theta = fminf(glm::dot(-unitDirection, record.normal), 1.0f);
  float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
  // 从高折射率到低折射率发出光线时是无法进行折射的会全反射
  bool cannot_refract = ri * sin_theta > 1.0f;
  glm::vec3 direction;

  if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(randState))
  {
    direction = reflect(unitDirection, record.normal);
  }
  else
  {
    direction = refract(unitDirection, record.normal, ri);
  }

  scatterRay = Ray(record.point, direction, ray.renderTime());

  return true;
}