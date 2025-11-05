#include "Camera.cuh"
#include "Hitable.cuh"
#include "HitableList.cuh"
#include "Ray.cuh"
#include "RayTracer.cuh"

#include "Dispatcher.hpp"
#include "Sphere.cuh"
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "Material.cuh"
#include "Metal.cuh"
#include "Lambertian.cuh"
#include "FuzzyMetalReflection.cuh"
#include "Dielectric.cuh"

#define HITABLE_SIZE 5
namespace Kernel
{
__device__ Material::Color color(curandState* state, const Ray& r, Hitable** dWorld)
{
  Ray currentRay = r;
  HitRecord record;
  Material::Color color(1.0f);
 
  for (int i = 0; i < 50; ++i)
  {
    // 将最小值设置为0.001避免击中点进入到surface内部
    if ((*dWorld)->hit(currentRay, 0.001, FLT_MAX, record))
    {
      Material::Color albedo;
      Ray scatteredRay;
      if (record.material->scatter(state, currentRay, record, albedo, scatteredRay))
      {
        color *= albedo;
        currentRay = scatteredRay;
      }
      else
      {
        return Material::Color(0.0f, 0.0f, 0.0f);
      }
    }
    else
    {
      glm::vec3 uDirection = glm::normalize(currentRay.direction());
      float t = 0.5f * (uDirection.y + 1.0f);
      Material::Color background = ((1 - t) * glm::vec3(1.0f, 1.0f, 1.0f) + t * glm::vec3(0.5f, 0.7f, 1.0f));
      return color * background;
    }
  }

  return Material::Color(0.0f, 0.0f, 0.0f);
  
}

__global__ void createWorld(Hitable** dList, Hitable** dWorld)
{
  dList[0] = new Sphere(glm::vec3(0,0,-1), 0.5,
                               new FuzzyMetalReflection(glm::vec3(0.8, 0.3, 0.3), .1));
  dList[1] = new Sphere(glm::vec3(0,-100.5,-1), 100,
                         new Metal(glm::vec3(0.8, 0.8, 0.0)));
  dList[2] = new Sphere(glm::vec3(1,0,-1), 0.5,
                         new Lambertian(glm::vec3(0.8, 0.6, 0.2)));
  dList[3] = new Sphere(glm::vec3(-1,0,-1), 0.5,
                       new Dielectric(1.5f));
  dList[4] = new Sphere(glm::vec3(-1,0,-1), 0.4,
                         new Dielectric(1.0f/1.5f));
  *dWorld = new HitableList(dList, HITABLE_SIZE);
}

__global__ void destroyWorld(Hitable** dList, Hitable** dWorld)
{
  for (int i = 0; i < HITABLE_SIZE; ++i)
  {
    delete *(dList + i);
  }
  delete *dWorld;
}

__global__ void renderInternal(Camera camera, ImageInfo imageInfo, glm::vec3 rayOrigin,
  Hitable** dWorld, int nsize, curandState* states)
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
    float x = (xIndex + curand_uniform(&states[randIndex])) / imageInfo.width;
    float y = (yIndex + curand_uniform(&states[randIndex])) / imageInfo.height;
    Ray ray = camera.getRay(x, y, &states[randIndex]);
    c += color(&states[randIndex], ray, dWorld);
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
  std::shared_ptr<Camera> mCamera;
  Hitable** dList;
  Hitable** dWorld;
  curandState* d_rand_state;
  ImageInfo mImageInfo;

  Impl()
    : mPBOResource(nullptr)
    , mImageDeviceId(nullptr)
    , mResourceSize(0)
    , d_rand_state(nullptr)
  {
    mCamera = std::make_shared<Camera>(glm::vec3(0.0f, 0.0f, 0.7f), glm::vec3(0.0f, 0.0f, -1.0f), 90.0f, 1920.0 / 1080);

    cudaMalloc(&dList, sizeof(Hitable*) * HITABLE_SIZE);
    cudaMalloc(&dWorld, sizeof(Hitable*));
    createWorld<<<1, 1>>>(dList, dWorld);
  }
  ~Impl()
  {
    destroyWorld<<<1, 1>>>(dList, dWorld);
    cudaFree(dList);
    cudaFree(dWorld);
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
void RayTracer::updateImage(const ImageInfo& imageInfo)
{
  if (!mImpl->mPBOResource)
  {
    return;
  }

  cudaGraphicsMapResources(1, &mImpl->mPBOResource, nullptr);
  cudaGraphicsResourceGetMappedPointer(
    reinterpret_cast<void**>(&mImpl->mImageDeviceId), &mImpl->mResourceSize, mImpl->mPBOResource);

  ImageInfo cImageInfo = imageInfo;
  cImageInfo.mColor = mImpl->mImageDeviceId;
  dim3 blockSize(8, 8, 1);
  dim3 gridSize((imageInfo.width + 7) / 8, (imageInfo.height + 7) / 8, 1);
  if (imageInfo.width != mImpl->mImageInfo.width || imageInfo.height != mImpl->mImageInfo.height)
  {
    float aspect = imageInfo.width * 1.0f / imageInfo.height;
    mImpl->mCamera->setAspect(aspect);
    cudaMalloc(&mImpl->d_rand_state, sizeof(curandState) * imageInfo.width * imageInfo.height);
    initRandom<<<gridSize, blockSize>>>(imageInfo.width, imageInfo.height, mImpl->d_rand_state);
    mImpl->mImageInfo = imageInfo;
  }
  // 抗锯齿参数
  int nSize = 64;

  renderInternal<<<gridSize, blockSize>>>(*mImpl->mCamera, cImageInfo,
    mImpl->mCamera->getCameraOrigin(), mImpl->dWorld, nSize, mImpl->d_rand_state);

  cudaDeviceSynchronize();
  cudaGraphicsUnmapResources(1, &mImpl->mPBOResource, nullptr);
}
Dispatcher::ObserverPtr RayTracer::getCamera() const
{
  return mImpl->mCamera;
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
