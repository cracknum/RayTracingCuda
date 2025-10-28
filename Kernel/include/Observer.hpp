#ifndef OBSERVER_CXX_H
#define OBSERVER_CXX_H
#include "KernelExports.hpp"
class QInputEvent;

class KERNEL_API Observer
{
public:
  virtual ~Observer() = default;
  virtual void OnMousePressed(const QInputEvent* event) = 0;
  virtual void onMouseRelease(const QInputEvent* event) = 0;
  virtual void onMouseMove(const QInputEvent* event) = 0;
  virtual void onWheelEvent(const QInputEvent* event) = 0;
  virtual void onKeyPressed(const QInputEvent* event) = 0;
  virtual void onKeyReleased(const QInputEvent* event) = 0;
};
#endif