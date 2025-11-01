#include "Camera.cuh"
#define _USE_MATH_DEFINES
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <math.h>
#include <qevent.h>

__host__ __device__ Camera::Camera(const glm::vec3& origin, float vfov, float aspect)
  : mOrigin(origin)
  , mUp(0.0f, 1.0f, 0.0f)
  , mForward(glm::vec3(0.0f, 0.0f, -1.0f))
  , mOrientation(1.0f, 0.0f, 0.0f, 0.0f)
  , mSpeed(0.01f)
  , mYaw(0.0f)
  , mPitch(0.0f)
  , mRoll(0.0f)
  , mVFOV(vfov)
  , mAspect(aspect)
  , mLeftButtonPressed(false)
  , mFocalLength(1.0f)
  , mRotateCenter(0.0f)
{
  mRotateLength = glm::length(mOrigin - mRotateCenter);
  setAspect(aspect);

  updateOrientation();
  updateCameraDirection();
}

void Camera::OnMousePressed(const QInputEvent* event)
{
  const auto* mouseEvent = static_cast<const QMouseEvent*>(event);
  QPoint mousePoint = mouseEvent->pos();

  mCurrentMousePos = glm::vec2(mousePoint.x(), mousePoint.y());
}

void Camera::onMouseRelease(const QInputEvent* event)
{
  mLeftButtonPressed = false;
}

void Camera::update()
{
  updateOrientation();
  updateCameraDirection();
  updateSpaceImageInformation();
}
void Camera::onMouseMove(const QInputEvent* event)
{
  const auto* mouseEvent = static_cast<const QMouseEvent*>(event);
  QPoint mousePoint = mouseEvent->pos();
  glm::vec2 moveDelta(0.0f);
  moveDelta.x = mousePoint.x() - mCurrentMousePos.x;
  moveDelta.y = mousePoint.y() - mCurrentMousePos.y;
  mCurrentMousePos = glm::vec2(mousePoint.x(), mousePoint.y());

  // 翻转y轴变化
  moveDelta.y *= -1;
  mYaw += moveDelta.x * mSpeed;
  mPitch += moveDelta.y * mSpeed;
  mPitch = glm::clamp(mPitch, -89.0f, 89.0f);

  update();
}

void Camera::onWheelEvent(const QInputEvent* event)
{
  auto* wheelEvent = static_cast<const QWheelEvent*>(event);
  QPoint angleDelta = wheelEvent->angleDelta();
  if (angleDelta.y() > 0)
  {
    // forward
    mRotateLength -= mSpeed;
  }
  else
  {
    // backward
    mRotateLength += mSpeed;
  }
  update();
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
  mSpaceImageInfo.mHeight = mFocalLength * tan(theta / 2) * 2.0f;
  mSpaceImageInfo.mWidth = aspect * mSpaceImageInfo.mHeight;
  update();
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
  update();
}
void Camera::moveToRight()
{
  mOrigin -= mSpeed * mRight;
  update();
}
void Camera::moveToTop()
{
  mOrigin -= mSpeed * mUp;
  update();
}
void Camera::moveToBottom()
{
  mOrigin += mSpeed * mUp;
  update();
}
void Camera::updateOrientation()
{
  glm::quat yaw = glm::angleAxis(mYaw, glm::vec3(0.0f, 1.0f, 0.0f));
  glm::quat pitch = glm::angleAxis(mPitch, glm::vec3(1.0f, 0.0f, 0.0f));
  mOrientation = pitch * yaw;
  mOrientation = glm::normalize(mOrientation);
}
void Camera::updateCameraDirection()
{
  mForward = mOrientation * glm::vec3(0.0f, 0.0f, -1.0f);
  mUp = mOrientation * glm::vec3(0.0f, 1.0f, 0.0f);
  mRight = mOrientation * glm::vec3(1.0f, 0.0f, 0.0f);
  mOrigin = mRotateCenter - mRotateLength * mForward;
}
void Camera::updateSpaceImageInformation()
{
  glm::vec3 forward = mOrientation * glm::vec3(0, 0, -1);
  glm::vec3 up = mOrientation * glm::vec3(0, 1, 0);
  glm::vec3 right = mOrientation * glm::vec3(1, 0, 0);

  glm::vec3 centerPoint = mOrigin + forward * mFocalLength;
  mSpaceImageInfo.mLowerLeftCorner =
    centerPoint - right * mSpaceImageInfo.mWidth * 0.5f - up * mSpaceImageInfo.mHeight * 0.5f;
  mSpaceImageInfo.mHorizontal = right * mSpaceImageInfo.mWidth;
  mSpaceImageInfo.mVertical = up * mSpaceImageInfo.mHeight;
}
