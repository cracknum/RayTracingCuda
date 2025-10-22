#ifndef RENDER_WINDOW_OPENGL_H
#define RENDER_WINDOW_OPENGL_H
#include "RenderExports.hpp"
#include <memory>
#include <qopenglfunctions_4_4_core.h>
#include <QOpenGLWidget>

class Element;

class RENDER_API RenderWindow : public QOpenGLWidget, protected QOpenGLFunctions_4_4_Core
{

public:
    explicit RenderWindow();
    ~RenderWindow();

    void addElement(Element* elem);
protected:
    virtual void initializeGL() override;
    virtual void resizeGL(int w, int h) override;
    virtual void paintGL() override;
    void initializeElements();

    private:
    struct Impl;
    std::unique_ptr<Impl> mImpl;
};
#endif /*RENDER_WINDOW_OPENGL_H*/
