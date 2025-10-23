#include <fstream>
#include <iostream>
#include <QApplication>
#include <sstream>

#include "RenderWindow.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "Element.hpp"
#include "Image.hpp"

#include "ImageMaker.h"
#include "Render.h"

std::string makeImage(int width, int height) {
  std::string image;
  image.resize(width * height * 3);

  for (size_t i = 0; i < width; i++) {
    for (int j = 0; j < height; j++) {
      float x = static_cast<float>(i) / width;
      float y = static_cast<float>(j) / height;

      int red = static_cast<int>(x * 255);
      int green = static_cast<int>(y * 255);
      int blue = static_cast<int>(x * y * 255);

      size_t index = (j * width + i) * 3;

      image[index] = red;
      image[index + 1] = green;
      image[index + 2] = blue;

      std::cout << i << " " << j << std::endl;
    }
  }

  return image;
}

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

  Kernel::render(imageInfo, spaceImageInfo, rayOrigin);
  
  image->setImage(width, height, imageInfo.mColor);

  RenderWindow window;
  window.resize(width, height);
  window.show();

  window.addElement(image);
  return app.exec();
}