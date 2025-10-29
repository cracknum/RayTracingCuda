#include "Camera.cuh"
#define _USE_MATH_DEFINES
#include <iostream>
#include <math.h>
#include <qevent.h>

__host__ __device__ Camera::Camera(
  const glm::vec3& lookAt, const glm::vec3& lookFrom, const glm::vec3& up, float vfov, float aspect)
  : mLookAt(lookAt)
  , mLookFrom(lookFrom)
  , mUp(up)
  , mVFOV(vfov)
  , mAspect(aspect)
  , mSpeed(0.1)
{
 setAspect(aspect);

  mOrigin = lookFrom;
  glm::vec3 w = glm::normalize(lookAt - lookFrom);
  glm::vec3 u = glm::normalize(up);
  mRight = glm::cross(w, u);
}

void Camera::OnMousePressed(const QInputEvent* event)
{
  std::cout << "mouse pressed" << std::endl;
}

void Camera::onMouseRelease(const QInputEvent* event)
{
  std::cout << "mouse release" << std::endl;
}

void Camera::onMouseMove(const QInputEvent* event)
{
  std::cout << "mouse move" << std::endl;
}

void Camera::onWheelEvent(const QInputEvent* event)
{
  std::cout << "wheel event" << std::endl;
}
void Camera::onKeyPressed(const QInputEvent* event)
{
  auto* keyEvent = static_cast<const QKeyEvent*>(event);
  if (keyEvent->key() == Qt::Key_W)
  {
    moveToTop();
  }
  else if (keyEvent->key() == Qt::Key_D)
  {
    moveToRight();
  }
  else if (keyEvent->key() == Qt::Key_A)
  {
    moveToLeft();
  }
  else if (keyEvent->key() == Qt::Key_S)
  {
    moveToBottom();
  }
}
void Camera::onKeyReleased(const QInputEvent* event) {}
Ray Camera::getRay(const float x, const float y) const
{
  Ray ray(mOrigin,
    mSpaceImageInfo.mLowerLeftCorner + x * mSpaceImageInfo.mHorizontal +
      y * mSpaceImageInfo.mVertical);

  return ray;
}
void Camera::setAspect(const float aspect)
{
  mAspect = aspect;

  float theta = mVFOV * M_PI / 180;
  float height = tan(theta / 2) * 2.0f;
  float width = aspect * height;
  mSpaceImageInfo.mLowerLeftCorner = glm::vec3(-2.0f, -1.0f, -1.0f);
  mSpaceImageInfo.mHorizontal = glm::vec3(width, 0.0f, 0.0f);
  mSpaceImageInfo.mVertical = glm::vec3(0.0f, height, 0.0f);
}
SpaceImageInfo Camera::getSpaceImageInfo() const
{
  return mSpaceImageInfo;
}
glm::vec3 Camera::getCameraOrigin() const
{
  return mOrigin;
}
void Camera::moveToLeft()
{
  mOrigin += mSpeed * mRight;
}
void Camera::moveToRight()
{
  mOrigin -= mSpeed * mRight;
}
void Camera::moveToTop()
{
  mOrigin -= mSpeed * mUp;
}
void Camera::moveToBottom()
{
  mOrigin += mSpeed * mUp;
}
