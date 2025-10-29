#ifndef RAYTRACER_CUDA_H
#define RAYTRACER_CUDA_H
#include "Dispatcher.hpp"
#include "KernelExports.hpp"
#include <glm/glm.hpp>
#include <memory>
#include <qopenglcontext.h>

struct SpaceImageInfo;
class Camera;

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

  Dispatcher::ObserverPtr getCamera() const;

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
}

#endif