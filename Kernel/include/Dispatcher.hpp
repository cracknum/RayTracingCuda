#ifndef DISPATCHER_CXX_H
#define DISPATCHER_CXX_H
#include "KernelExports.hpp"
#include <memory>
#include <QInputEvent>
#include "Observer.hpp"

class KERNEL_API Dispatcher
{
public:
  using ObserverPtr = std::shared_ptr<Observer>;
  Dispatcher();
  ~Dispatcher();

  void addObserver(const ObserverPtr& observer);

  void handle(const QInputEvent* event);

protected:
  void handle(const QInputEvent* event, ObserverPtr& observer);

private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
#endif