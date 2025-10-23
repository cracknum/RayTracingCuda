#include <fstream>
#include <iostream>
#include <QApplication>
#include <sstream>

#include "RenderWindow.hpp"
#ifdef ENABLE_IMAGE_DEBUG
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#endif

#include "Image.hpp"
#include "RayTracer.cuh"

int main(int argc, char **argv) {
  QApplication app(argc, argv);
  Image *image = new Image();

  int width = 1920;
  int height = 1080;

  Kernel::ImageInfo imageInfo;
  imageInfo.width = width;
  imageInfo.height = height;
  imageInfo.mColor = new unsigned char[imageInfo.width * imageInfo.height * 3]{0};

  Kernel::SpaceImageInfo spaceImageInfo;
  spaceImageInfo.mLowerLeftCorner = glm::vec3(-2.0f, -1.0f, -1.0f);
  spaceImageInfo.mHorizontal = glm::vec3(4.0f, 0.0f, 0.0f);
  spaceImageInfo.mVertical = glm::vec3(0.0f, 2.0f, 0.0f);
  glm::vec3 rayOrigin(0.0f, 0.0f, 0.0f);

  Kernel::RayTracer rayTracer;
  rayTracer.render(imageInfo, spaceImageInfo, rayOrigin);
  
  image->setImage(width, height, imageInfo.mColor);

  RenderWindow window;
  window.resize(width, height);
  window.show();

  window.addElement(image);
  return app.exec();
}