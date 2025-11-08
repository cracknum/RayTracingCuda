#ifndef TEXTURE_LOADER_H
#define TEXTURE_LOADER_H
#include <cuda_runtime.h>
#include <string>

template<typename ComponentType>
struct Image;
class TextureLoader {
public:
  static TextureLoader* getInstance();
  cudaTextureObject_t getTexture(const std::string& path);
protected:
  TextureLoader() = default;

  Image<unsigned char> loadImage(const std::string& path);
};



#endif // TEXTURE_LOADER_H
