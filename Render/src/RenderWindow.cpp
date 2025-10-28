#include "RenderWindow.hpp"
#include "Element.hpp"
#include <QColor>
#include <QVector4D>
#include "Dispatcher.hpp"
#include "Camera.cuh"
struct RenderWindow::Impl final
{
  QColor mClearColor;
  std::vector<std::unique_ptr<Element>> mElements;
  std::unique_ptr<Dispatcher> mDispatcher;
  std::shared_ptr<Observer> mCamera;

  Impl()
    : mClearColor(16, 48, 72, 1)
  {
    mDispatcher = std::make_unique<Dispatcher>();
    mCamera = std::shared_ptr<Observer>(new Camera(glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f), glm::vec3(0.f, 0.f, 0.f), 90, 0.9f));
    mDispatcher->addObverser(mCamera);
  }
};

RenderWindow::RenderWindow()
{
  mImpl = std::make_unique<Impl>();
}

RenderWindow::~RenderWindow() {}

void RenderWindow::addElement(Element* elem)
{
  mImpl->mElements.push_back(std::unique_ptr<Element>(elem));

  if (context())
  {
    initializeElements();
  }
}

void RenderWindow::initializeGL()
{
  this->initializeOpenGLFunctions();
  glClearColor(mImpl->mClearColor.redF(), mImpl->mClearColor.greenF(), mImpl->mClearColor.blueF(),
    mImpl->mClearColor.alphaF());
  glClear(GL_COLOR_BUFFER_BIT);
  initializeElements();
}

void RenderWindow::resizeGL(int w, int h)
{
  this->glViewport(0, 0, w, h);
}

void RenderWindow::paintGL()
{
  for (auto& elem : mImpl->mElements)
  {
    if (!elem->initialized())
    {
      elem->initialize(this);
    }
    elem->render();
  }
}

void RenderWindow::initializeElements()
{
  for (auto& elem : mImpl->mElements)
  {
    if (!elem->initialized())
    {
      elem->initialize(this);
    }
  }
}

void RenderWindow::mousePressEvent(QMouseEvent* event)
{
  mImpl->mDispatcher->handle(event);
}

void RenderWindow::mouseReleaseEvent(QMouseEvent* event)
{
  mImpl->mDispatcher->handle(event);
}

void RenderWindow::mouseMoveEvent(QMouseEvent* event)
{
  mImpl->mDispatcher->handle(event);
}

void RenderWindow::wheelEvent(QWheelEvent* event)
{
  mImpl->mDispatcher->handle(event);
}
void RenderWindow::keyPressEvent(QKeyEvent* event)
{
  mImpl->mDispatcher->handle(event);
}
void RenderWindow::keyReleaseEvent(QKeyEvent* event)
{
  mImpl->mDispatcher->handle(event);
}
