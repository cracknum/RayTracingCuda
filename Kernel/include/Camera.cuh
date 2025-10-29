#ifndef CAMERA_CUDA_H
#define CAMERA_CUDA_H
#include "KernelExports.hpp"
#include "Observer.hpp"
#include <glm/detail/type_quat.hpp>
#include <glm/glm.hpp>
#include "Ray.cuh"

struct KERNEL_API SpaceImageInfo
{
  glm::vec3 mLowerLeftCorner;
  glm::vec3 mHorizontal;
  glm::vec3 mVertical;
};

class QInputEvent;
class QPoint;

class KERNEL_API Camera : public Observer
{
public:
  __host__ __device__
  Camera() = default;
  // 引入平移，旋转（四元数）
  __host__ __device__
  Camera(const glm::vec3& lookAt, const glm::vec3& lookFrom, const glm::vec3& up, float vfov,
    float aspect);
  __device__
  Ray getRay(const float x, const float y) const;

  void setAspect(const float aspect);
  SpaceImageInfo getSpaceImageInfo() const;
  glm::vec3 getCameraOrigin() const;

  void OnMousePressed(const QInputEvent* event) override;
  void onMouseRelease(const QInputEvent* event) override;
  void onMouseMove(const QInputEvent* event) override;
  void onWheelEvent(const QInputEvent* event) override;

  void onKeyPressed(const QInputEvent* event) override;
  void onKeyReleased(const QInputEvent* event) override;
protected:
  void moveToLeft();
  void moveToRight();
  void moveToTop();
  void moveToBottom();

private:
  glm::vec3 mLookAt;
  glm::vec3 mLookFrom;
  glm::vec3 mUp;
  glm::vec3 mOrigin;
  glm::vec3 mRight;
  glm::vec3 mForward;
  float mSpeed;

  // 垂直视场角
  float mVFOV;
  // 在使用垂直视场角计算完view plan的高度后通过乘上aspect得到宽度, aspect = width/height
  float mAspect;

  glm::quat mTransform;
  SpaceImageInfo mSpaceImageInfo;
};
#endif