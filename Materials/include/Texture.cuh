#ifndef TEXTURE_CUH
#define TEXTURE_CUH
#include <cuda_runtime.h>
#include <glm/glm.hpp>

class Texture
{
public:
  using Color = glm::vec3;
  __device__ virtual ~Texture() = default;

  __device__ virtual Color value(float u, float v, const glm::vec3& point) = 0;
};

class SolidTexture : public Texture
{
public:
  __device__ explicit SolidTexture(const Color& color)
    : mColor(color)
  {
  }
  __device__ __forceinline__ Color value(float u, float v, const glm::vec3& point) override
  {
    return mColor;
  }

private:
  Color mColor;
};

#endif // TEXTURE_CUH
