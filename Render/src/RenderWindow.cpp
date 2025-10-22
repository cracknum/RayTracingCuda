#include "Element.hpp"
#include "RenderWindow.hpp"
#include <QColor>
#include <QVector4D>
struct RenderWindow::Impl final
{
    QColor mClearColor;
    std::vector<std::unique_ptr<Element>> mElements;

    Impl()
    : mClearColor(16, 48, 72, 1)
    {
    }
};

RenderWindow::RenderWindow()
{
    mImpl = std::make_unique<Impl>();
}

RenderWindow::~RenderWindow()
{
}

void RenderWindow::addElement(Element* elem)
{
    mImpl->mElements.push_back(std::unique_ptr<Element>(elem));

    if (context()) {
      initializeElements();
    }
}

void RenderWindow::initializeGL()
{
    this->initializeOpenGLFunctions();
    glClearColor(mImpl->mClearColor.redF(), mImpl->mClearColor.greenF(), mImpl->mClearColor.blueF(), mImpl->mClearColor.alphaF());
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

void RenderWindow::initializeElements() {
  for (auto& elem : mImpl->mElements) {
    if (!elem->initialized()) {
      elem->initialize(this);
    }
  }
}
