#ifndef IMAGE_OPENGL_H
#define IMAGE_OPENGL_H
#include "Element.hpp"
#include "RenderExports.hpp"
#include <cstdint>
#include <memory>

class QOpenGLFunctions_4_4_Core;

class RENDER_API Image: public Element
{
public:
    using SuperClass = Element;
    using Self = Image;
    Image();
    ~Image() override;
    void render() override;
    
    void setImage(unsigned int width, unsigned int height);
    void initialize(QOpenGLFunctions_4_4_Core* gl) override;
    
    unsigned int getImageTexture();
    protected:
    void drawOnImage();
    void bind();
    void unbind();

    void uploadImage();

   private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
#endif // IMAGE_OPENGL_H