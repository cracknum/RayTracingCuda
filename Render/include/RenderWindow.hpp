#ifndef RENDER_WINDOW_OPENGL_H
#define RENDER_WINDOW_OPENGL_H
#include "RenderExports.hpp"
#include <QOpenGLWidget>
#include <memory>
#include <qopenglfunctions_4_4_core.h>

class Element;
class QMouseEvent;

class RENDER_API RenderWindow
  : public QOpenGLWidget
  , protected QOpenGLFunctions_4_4_Core
{

public:
  explicit RenderWindow();
  ~RenderWindow();

  void addElement(Element* elem);

protected:
  void initializeGL() override;
  void resizeGL(int w, int h) override;
  void paintGL() override;
  void initializeElements();

  void mousePressEvent(QMouseEvent* event) override;
  void mouseReleaseEvent(QMouseEvent* event) override;
  void mouseMoveEvent(QMouseEvent* event) override;
  void wheelEvent(QWheelEvent* event) override;

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
#endif /*RENDER_WINDOW_OPENGL_H*/
