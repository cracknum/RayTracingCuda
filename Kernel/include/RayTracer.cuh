#ifndef RAYTRACER_CUDA_H
#define RAYTRACER_CUDA_H
#include "KernelExports.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <qopenglcontext.h>

class SpaceImageInfo;

namespace Kernel
{
struct KERNEL_API ImageInfo
{
  unsigned char* mColor;
  int width;
  int height;
};

class KERNEL_API RayTracer
{
public:
  RayTracer();
  ~RayTracer();
  void bindImagePBO(GLuint pbo);
  void unbindImagePBO(GLuint pbo);
  void updateImage(const ImageInfo& imageInfo);

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
}

#endif