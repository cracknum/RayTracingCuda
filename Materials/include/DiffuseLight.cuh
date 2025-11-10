#ifndef DIFFUSE_LIGHT_CUDA_H
#define DIFFUSE_LIGHT_CUDA_H
#include "Material.cuh"
#include <cuda_runtime.h>

class DiffuseLight : public Material
{
public:
  __device__ __forceinline__ DiffuseLight(cudaTextureObject_t texture)
  {
    mTexture = texture;
  }
  __device__ __forceinline__ DiffuseLight(const Color& color)
  {
    mTexture = 0;
    mColor = color;
  }
  __device__ __forceinline__ virtual bool scatter(
    curandState* randState, const Ray& ray, const HitRecord& record, Color& color, Ray& scatterRay)
    {
        return false;
    }
  __device__ virtual Color emitted(float u, float v, const glm::vec3& point)
  {
        if (mTexture)
        {
            float4 c = tex2D<float4>(mTexture, u, v);
            return Color(c.x, c.y, c.z);
        }

        return mColor;
  }

  private:
  Color mColor;
};
#endif // DIFFUSE_LIGHT_CUDA_H