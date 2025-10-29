#ifndef ELEMENT_OPENGL_H
#define ELEMENT_OPENGL_H

#include <fstream>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>

#include "RenderExports.hpp"
class QOpenGLFunctions_4_4_Core;
namespace Kernel
{
class RayTracer;
}
class RENDER_API Element
{
public:
  Element()
    : mContext(nullptr)
    , mRayTracer(nullptr)
  {
  }
  virtual ~Element() = default;
  virtual void update() = 0;
  virtual void render() = 0;
  bool initialized() const { return mInitialized; };
  virtual void initialize(QOpenGLFunctions_4_4_Core* gl) { mContext = gl; }
  virtual void setRayTracer(Kernel::RayTracer* rayTracer) { mRayTracer = rayTracer; }

protected:
  std::string readFile(const std::string& filepath)
  {
    std::ifstream in(filepath);
    if (in.is_open())
    {
      std::stringstream ss;
      ss << in.rdbuf();
      return ss.str();
    }
    std::cout << "can't open " << filepath << std::endl;

    return std::string();
  }

protected:
  bool mInitialized = false;
  QOpenGLFunctions_4_4_Core* mContext;
  Kernel::RayTracer* mRayTracer;
};
#endif