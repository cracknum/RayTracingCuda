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

  // std::string &imageStr = makeImage(100, 100);
  // int error = stbi_write_jpg("./testImage.jpg", 540, 960, 3, imageStr.data(),
  // 0);
  // image->setImage(1920, 1080, imageStr);
  RenderWindow window;
  window.resize(1920, 1080);
  window.show();

  // window.addElement(image);
  return app.exec();
}