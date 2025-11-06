#include "Camera.cuh"
#define _USE_MATH_DEFINES
#include <glm/gtc/quaternion.hpp>
#include <iostream>
#include <math.h>
#include <qevent.h>

__host__ __device__ Camera::Camera(
  const glm::vec3& origin, const glm::vec3& viewPoint, float vfov, float aspect)
  : mOrigin(origin)
  , mUp(0.0f, 1.0f, 0.0f)
  , mOrientation(1.0f, 0.0f, 0.0f, 0.0f)
  , mSpeed(0.01f)
  , mYaw(0.0f)
  , mPitch(0.0f)
  , mRoll(0.0f)
  , mVFOV(vfov)
  , mAspect(aspect)
  , mLeftButtonPressed(false)
  , mFocalLength(1.0f)
  , mRotateCenter(viewPoint)
  , mDefocusAngle(0)
  , mFocusDistance(10.0f)
{
  mRotateLength = glm::length(mOrigin - mRotateCenter);
  mForward = glm::normalize(viewPoint - origin);
  mRight = mOrientation * glm::cross(mUp, mForward);
  setAspect(aspect);
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
  std::cout << "distance: " << glm::length(mOrigin) << std::endl;
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
Ray Camera::getRay(const float x, const float y, curandState* state) const
{
  auto origin = mOrigin;
  if (mDefocusAngle > 0)
  {
    origin = lensDiskSample(mOrigin, state);
  }

  Ray ray(origin,
    mSpaceImageInfo.mLowerLeftCorner + x * mSpaceImageInfo.mHorizontal +
      y * mSpaceImageInfo.mVertical, curand_uniform(state));

  return ray;
}
void Camera::setAspect(const float aspect)
{
  mAspect = aspect;

  float theta = mVFOV * M_PI / 180;
  mSpaceImageInfo.mHeight = mFocalLength * tan(theta / 2) * 2.0f * mFocusDistance;
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
  mRotateCenter -= mSpeed * mRight;
  update();
}
void Camera::moveToRight()
{
  mRotateCenter += mSpeed * mRight;
  update();
}
void Camera::moveToTop()
{
  mRotateCenter += mSpeed * mUp;
  update();
}
void Camera::moveToBottom()
{
  mRotateCenter -= mSpeed * mUp;
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
  glm::vec3 forward = mOrientation * mForward;
  mOrigin = mRotateCenter - mRotateLength * forward;
}
void Camera::updateSpaceImageInformation()
{
  glm::vec3 forward = mOrientation * mForward;
  glm::vec3 up = mOrientation * mUp;
  glm::vec3 right = mOrientation * glm::cross(mForward, mUp);

  glm::vec3 centerPoint = mOrigin + forward * mFocalLength;
  mSpaceImageInfo.mLowerLeftCorner = centerPoint + mFocusDistance * forward -
    right * mSpaceImageInfo.mWidth * 0.5f - up * mSpaceImageInfo.mHeight * 0.5f;
  mSpaceImageInfo.mHorizontal = right * mSpaceImageInfo.mWidth;
  mSpaceImageInfo.mVertical = up * mSpaceImageInfo.mHeight;

  // 更新透镜平面信息
  float radian = mDefocusAngle * M_PI / 180.0f;
  float defocusRadius = mFocusDistance * tanf(radian / 2);
  mLensXVector = right * defocusRadius;
  mLensYVector = up * defocusRadius;
}

__device__
glm::vec3 Camera::lensDiskSample(const glm::vec3& center,curandState* state) const
{
  glm::vec3 unitVector = randomInUnitDisk(state);

  return center + unitVector.x * mLensXVector + unitVector.y * mLensYVector;
}
__device__ glm::vec3 Camera::randomInUnitDisk(curandState* state) const
{
  while (true)
  {
    glm::vec3 randVec(
      curand_uniform(state) * 2.0f - 1.0f, curand_uniform(state) * 2.0f - 1.0f, 0.0f);
    if (glm::length(randVec) < 1)
    {
      return randVec;
    }
  }
}
