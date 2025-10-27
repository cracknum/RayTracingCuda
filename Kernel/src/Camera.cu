#include "Camera.cuh"
#define _USE_MATH_DEFINES
#include <math.h>

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
