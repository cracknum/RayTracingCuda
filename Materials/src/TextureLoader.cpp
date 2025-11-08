#include "TextureLoader.h"
#define STB_IMAGE_IMPLEMENTATION
#include <iostream>
#include <stb_image.h>
template <typename ComponentType>
struct Image
{
  int width;
  int height;
  int channels;
  ComponentType* data;

  Image()
    : width(0)
    , height(0)
    , channels(0)
    , data(nullptr)
  {
  }
  ~Image()
  {
    if (data)
    {
      delete data;
      width = 0;
      height = 0;
      channels = 0;
    }
  }
};
TextureLoader* TextureLoader::getInstance()
{
  static TextureLoader loader;
  return &loader;
}
cudaTextureObject_t TextureLoader::getTexture(const std::string& path)
{
  auto image = loadImage(path);
  cudaTextureObject_t texture;
  cudaArray_t array;
  cudaChannelFormatDesc format = cudaCreateChannelDesc<uchar4>();
  cudaMallocArray(&array, &format, image.width, image.height);

  int srcPitch = image.width * image.channels * sizeof(unsigned char);
  cudaMemcpy2DToArray(
    array, 0, 0, image.data, srcPitch, srcPitch, image.height, cudaMemcpyHostToDevice);

  cudaResourceDesc resource{};
  resource.resType = cudaResourceTypeArray;
  resource.res.array.array = array;
  cudaTextureDesc texture_desc{};
  texture_desc.addressMode[0] = cudaAddressModeWrap;
  texture_desc.addressMode[1] = cudaAddressModeWrap;
  texture_desc.filterMode = cudaFilterModeLinear;
  // 因为使用的是unsigned char，所以这里自动进行归一化，归一化除的最大值为255，因为255是unsigned char的最大值
  texture_desc.readMode = cudaReadModeNormalizedFloat;
  texture_desc.normalizedCoords = 1;

  cudaCreateTextureObject(&texture, &resource, &texture_desc, nullptr);
  return texture;
}
Image<unsigned char> TextureLoader::loadImage(const std::string& path)
{
  int width;
  int height;
  int channels;
  stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 4);
  Image<unsigned char> image;

  if (!data)
  {
    std::cout << stbi_failure_reason() << std::endl;
    return image;
  }

  image.width = width;
  image.height = height;
  image.channels = channels;
  image.data = new unsigned char[width * height * channels]{ 0 };
  memcpy(image.data, data, width * height * channels);
  return image;
}