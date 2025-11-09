#include "Lambertian.cuh"

#include "HitRecord.cuh"
#include "Ray.cuh"
Lambertian::Lambertian(const Color& albedo)
{
  mAlbedo = albedo;
  mTexture = 0;
}
Lambertian::Lambertian(cudaTextureObject_t texture)
{
  mTexture = texture;
}

__device__ bool Lambertian::scatter(curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
{
  glm::vec3 scatterDirection = ramdomHemisphere(randState, record.normal);
  if (isNearZero(scatterDirection))
  {
    scatterDirection = record.normal;
  }
  scatterRay = Ray(record.point, scatterDirection, ray.renderTime());
  if (mTexture)
  {
    auto c = tex2D<float4>(mTexture, record.u, record.v);
    color = glm::vec3(c.x, c.y, c.z);
  }
  else
  {
    color = mAlbedo;
  }


  return true;
}