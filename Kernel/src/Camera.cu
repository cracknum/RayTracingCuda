#include "Camera.cuh"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>

Camera::Camera(
  const glm::vec3& lookAt, const glm::vec3& lookFrom, const glm::vec3& up, float vfov, float aspect)
  : mLookAt(lookAt)
  , mLookFrom(lookFrom)
  , mUp(up)
  , mVFOV(vfov)
  , mAspect(aspect)
{
    // 角度转弧度
    float theta = vfov * M_PI / 180;
    float halfHeight = tan(theta / 2);
    float halfWidth = aspect * halfHeight;
    mOrigin = lookFrom;

    glm::vec3 w = glm::normalize(lookFrom - lookAt);
    glm::vec3 u = glm::normalize(up);
    glm::vec3 v = glm::cross(u, w);
}

void Camera::OnMousePressed(const QInputEvent* event) {
  std::cout << "mouse pressed" << std::endl;
}

void Camera::onMouseRelease(const QInputEvent* event) {
  std::cout << "mouse release" << std::endl;
}

void Camera::onMouseMove(const QInputEvent* event) {
  std::cout << "mouse move" << std::endl;
}

void Camera::onWheelEvent(const QInputEvent* event)
{
  std::cout << "wheel event" << std::endl;
}
void Camera::onKeyPressed(const QInputEvent* event)
{
  std::cout << "key pressed" << std::endl;
}
void Camera::onKeyReleased(const QInputEvent* event)
{
  std::cout << "key released" << std::endl;
}
