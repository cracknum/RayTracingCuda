#include "Hitable.cuh"
#include "HitableList.cuh"
#include "Ray.cuh"
#include "RayTracer.cuh"
#include "Sphere.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
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

__device__ glm::vec3 color(const Ray& r, Hitable** dWorld)
{
  HitRecord record;
  if ((*dWorld)->hit(r, 0, FLT_MAX, record))
  {
    return 0.5f * glm::vec3(record.normal.x + 1.0f, record.normal.y + 1.0f, record.normal.z + 1.0f);
  }
  else
  {
    glm::vec3 uDirection = glm::normalize(r.direction());
    float t = 0.5f * (uDirection.y + 1.0f);

    return (1 - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f);
  }
}

__global__ void createWorld(Hitable** dList, Hitable** dWorld)
{
  *dList = new Sphere(glm::vec3(0, 0, -1), 0.5);
  *(dList + 1) = new Sphere(glm::vec3(0, -100.5, -1), 100);
  *dWorld = new HitableList(dList, 2);
}

__global__ void destroyWorld(Hitable** dList, Hitable** dWorld)
{
  delete *dList;
  delete *(dList + 1);
  delete *dWorld;
}

__global__ void renderInternal(ImageInfo imageInfo, SpaceImageInfo spaceImageInfo,
  glm::vec3 rayOrigin, Hitable** dWorld, int nsize, curandState* states)
{
  int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
  int yIndex = blockDim.y * blockIdx.y + threadIdx.y;

  if (xIndex >= imageInfo.width || yIndex >= imageInfo.height)
  {
    return;
  }

  int pixelIndex = (yIndex * imageInfo.width + xIndex) * 3;
  int randIndex = yIndex * imageInfo.width + xIndex;

  glm::vec3 c(0, 0, 0);

  for (int i = 0; i < nsize; ++i)
  {
    float x = static_cast<float>(xIndex + curand_uniform(&states[randIndex])) /
      static_cast<float>(imageInfo.width);
    float y = static_cast<float>(yIndex + curand_uniform(&states[randIndex])) /
      static_cast<float>(imageInfo.height);

    Ray ray(rayOrigin,
      spaceImageInfo.mLowerLeftCorner + x * spaceImageInfo.mHorizontal +
        y * spaceImageInfo.mVertical - rayOrigin);
    c += color(ray, dWorld);
  }

  c /= static_cast<float>(nsize);

  imageInfo.mColor[pixelIndex] = c.x * 255;
  imageInfo.mColor[pixelIndex + 1] = c.y * 255;
  imageInfo.mColor[pixelIndex + 2] = c.z * 255;
}

__global__ void initRandom(int width, int height, curandState* states)
{
  int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index = yIndex * width + xIndex;

  if (xIndex >= width || yIndex >= height)
  {
    return;
  }

  curand_init(1984, index, 0, &states[index]);
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
void RayTracer::updateImage(
  ImageInfo& imageInfo, const SpaceImageInfo& spaceImageInfo, const glm::vec3& rayOrigin)
{
  if (!mImpl->mPBOResource)
  {
    return;
  }

  cudaGraphicsMapResources(1, &mImpl->mPBOResource, nullptr);
  cudaGraphicsResourceGetMappedPointer(
    reinterpret_cast<void**>(&mImpl->mImageDeviceId), &mImpl->mResourceSize, mImpl->mPBOResource);

  Hitable** dList;
  cudaMalloc(&dList, sizeof(Hitable*) * 2);
  Hitable** dWorld;
  cudaMalloc(&dWorld, sizeof(Hitable*));

  createWorld<<<1, 1>>>(dList, dWorld);

  curandState* d_rand_state;
  cudaMalloc(&d_rand_state, sizeof(curandState) * imageInfo.width * imageInfo.height);

  ImageInfo cImageInfo = imageInfo;
  cImageInfo.mColor = mImpl->mImageDeviceId;
  dim3 blockSize(8, 8, 1);
  dim3 gridSize((imageInfo.width + 7) / 8, (imageInfo.height + 7) / 8, 1);

  initRandom<<<gridSize, blockSize>>>(imageInfo.width, imageInfo.height, d_rand_state);

  // 抗锯齿参数
  int nSize = 1;

  renderInternal<<<gridSize, blockSize>>>(
    cImageInfo, spaceImageInfo, rayOrigin, dWorld, nSize, d_rand_state);

  cudaDeviceSynchronize();

  destroyWorld<<<1, 1>>>(dList, dWorld);
  cudaFree(dList);
  cudaFree(dWorld);
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
