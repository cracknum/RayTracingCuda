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
  auto image = new Image();

  int width = 1920;
  int height = 1080;
  image->setImage(width, height);

  RenderWindow window;
  window.resize(width, height);
  window.show();

  window.addElement(image);
  return app.exec();
}