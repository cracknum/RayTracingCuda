#include "Ray.cuh"
#include "RayTracer.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
namespace Kernel
{
__device__ bool hitSphere(const glm::vec3& center, float radius, const Ray& r)
{
  glm::vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = 2.0f * dot(oc, r.direction());
  float c = dot(oc, oc) - radius * radius;
  float discriminant = b * b - 4.0f * a * c;
  return (discriminant > 0.0f);
}

__device__ glm::vec3 color(const Ray& r)
{
  if (hitSphere(glm::vec3(0, 0, -1), 0.5, r))
  {
    return glm::vec3(1.0f, 0.0f, 0.0f);
  }

  glm::vec3 uDirection = glm::normalize(r.direction());
  float t = 0.5f * (uDirection.y + 1.0f);

  return (1 - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
}

__global__ void renderInternal(
  ImageInfo imageInfo, SpaceImageInfo spaceImageInfo, glm::vec3 rayOrigin)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  if (xIndex >= imageInfo.width || yIndex >= imageInfo.height)
  {
    return;
  }

  int pixelIndex = (yIndex * imageInfo.width + xIndex) * 3;

  float x = static_cast<float>(xIndex) / static_cast<float>(imageInfo.width);
  float y = static_cast<float>(yIndex) / static_cast<float>(imageInfo.height);

  Ray ray(rayOrigin,
    spaceImageInfo.mLowerLeftCorner + x * spaceImageInfo.mHorizontal +
      y * spaceImageInfo.mVertical);
  glm::vec3 c = color(ray);
  imageInfo.mColor[pixelIndex] = c.x * 255;
  imageInfo.mColor[pixelIndex + 1] = c.y * 255;
  imageInfo.mColor[pixelIndex + 2] = c.z * 255;
}

void RayTracer::render(
  ImageInfo& imageInfo, const SpaceImageInfo& spaceImageInfo, const glm::vec3& rayOrigin)
{
  unsigned char* devId;
  int imageSize = imageInfo.width * imageInfo.height * sizeof(unsigned char) * 3;
  cudaMalloc(&devId, imageSize);
  cudaMemset(devId, 0, imageSize);
  cudaMemcpy(devId, imageInfo.mColor, imageSize, cudaMemcpyHostToDevice);

  ImageInfo cImageInfo = imageInfo;
  cImageInfo.mColor = devId;
  dim3 threadSize(8, 8, 1);
  dim3 blockSize((imageInfo.width + 7) / 8, (imageInfo.height + 7) / 8, 1);

  renderInternal<<<blockSize, threadSize>>>(cImageInfo, spaceImageInfo, rayOrigin);

  cudaDeviceSynchronize();
  cudaMemcpy(imageInfo.mColor, cImageInfo.mColor, imageSize, cudaMemcpyDeviceToHost);

  cudaFree(cImageInfo.mColor);
}

struct RayTracer::Impl
{
  cudaGraphicsResource* mPBOResource;
  unsigned char* mImageDeviceId;
  size_t mResourceSize;
  Impl()
    : mPBOResource(nullptr)
    , mImageDeviceId(nullptr)
    , mResourceSize(0)
  {
  }
};

void RayTracer::bindImagePBO(GLuint pbo)
{
  cudaGraphicsGLRegisterBuffer(&mImpl->mPBOResource, pbo, cudaGLMapFlagsWriteDiscard);
}

void RayTracer::unbindImagePBO(GLuint pbo)
{
  cudaGraphicsUnregisterResource(mImpl->mPBOResource);
}
void RayTracer::updateImage(ImageInfo& imageInfo, const SpaceImageInfo& spaceImageInfo, const glm::vec3& rayOrigin)
{
  if (!mImpl->mPBOResource)
  {
    return;
  }

  cudaGraphicsMapResources(1, &mImpl->mPBOResource, nullptr);

  cudaGraphicsResourceGetMappedPointer(
    reinterpret_cast<void**>(&mImpl->mImageDeviceId), &mImpl->mResourceSize, mImpl->mPBOResource);

  int imageSize = imageInfo.width * imageInfo.height * sizeof(unsigned char) * 3;

  ImageInfo cImageInfo = imageInfo;
  cImageInfo.mColor = mImpl->mImageDeviceId;
  dim3 threadSize(8, 8, 1);
  dim3 blockSize((imageInfo.width + 7) / 8, (imageInfo.height + 7) / 8, 1);

  renderInternal<<<blockSize, threadSize>>>(cImageInfo, spaceImageInfo, rayOrigin);

  cudaDeviceSynchronize();

  cudaGraphicsUnmapResources(1, &mImpl->mPBOResource, nullptr);

}

RayTracer::RayTracer()
{
  mImpl = std::make_unique<Impl>();
}

RayTracer::~RayTracer()
{
  if (mImpl->mPBOResource)
  {
    cudaGraphicsUnmapResources(1, &mImpl->mPBOResource, nullptr);
    cudaGraphicsUnregisterResource(mImpl->mPBOResource);
    mImpl->mPBOResource = nullptr;
    mImpl->mImageDeviceId = nullptr;
    mImpl->mResourceSize = 0;
  }
}
} // namespace Kernel
