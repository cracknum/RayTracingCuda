#ifndef RAYTRACER_CUDA_H
#define RAYTRACER_CUDA_H
#include "KernelExports.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <qopenglcontext.h>

namespace Kernel
{
struct KERNEL_API ImageInfo
{
  unsigned char* mColor;
  int width;
  int height;
};

struct KERNEL_API SpaceImageInfo
{
  glm::vec3 mLowerLeftCorner;
  glm::vec3 mHorizontal;
  glm::vec3 mVertical;
};

class KERNEL_API RayTracer
{
public:
  RayTracer();
  ~RayTracer();
  void bindImagePBO(GLuint pbo);
  void unbindImagePBO(GLuint pbo);
  void updateImage(ImageInfo& imageInfo, const SpaceImageInfo& spaceImageInfo, const glm::vec3& rayOrigin);

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
}

#endif