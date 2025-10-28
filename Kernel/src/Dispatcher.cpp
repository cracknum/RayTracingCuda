#include "Dispatcher.hpp"
#include "Observer.hpp"
#include <QMouseEvent>
#include <algorithm>
#include <vector>

struct Dispatcher::Impl
{
  std::vector<ObserverPtr> mObversers;
};

Dispatcher::Dispatcher()
{
  mImpl = std::make_unique<Impl>();
}

Dispatcher::~Dispatcher() {}

void Dispatcher::handle(const QInputEvent* event)
{
  std::for_each(mImpl->mObversers.begin(), mImpl->mObversers.end(),
    [event, this](ObserverPtr& observer) { this->handle(event, observer); });
}

void Dispatcher::handle(const QInputEvent* event, ObserverPtr& observer)
{
  if (event->type() == QEvent::MouseButtonPress)
  {
    observer->OnMousePressed(event);
  }
  else if (event->type() == QEvent::MouseButtonRelease)
  {
    observer->onMouseRelease(event);
  }
  else if (event->type() == QEvent::MouseMove)
  {
    observer->onMouseMove(event);
  }
  else if (event->type() == QEvent::Wheel)
  {
    observer->onWheelEvent(event);
  }
}

void Dispatcher::addObverser(const ObserverPtr& observer)
{
  mImpl->mObversers.push_back(observer);
}
