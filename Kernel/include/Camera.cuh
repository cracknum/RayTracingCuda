#ifndef CAMERA_CUDA_H
#define CAMERA_CUDA_H
#include "KernelExports.hpp"
#include "Observer.hpp"
#include "Ray.cuh"
#include <glm/detail/type_quat.hpp>
#include <glm/glm.hpp>

struct KERNEL_API SpaceImageInfo
{
  float mWidth;
  float mHeight;
  glm::vec3 mLowerLeftCorner;
  glm::vec3 mHorizontal;
  glm::vec3 mVertical;
};

class QInputEvent;
class QPoint;

class KERNEL_API Camera : public Observer
{
public:
  __host__ __device__ Camera() = default;
  // 引入平移，旋转（四元数）
  __host__ __device__ Camera(const glm::vec3& origin, float vfov, float aspect);
  __device__ Ray getRay(const float x, const float y) const;

  void setAspect(const float aspect);
  SpaceImageInfo getSpaceImageInfo() const;
  glm::vec3 getCameraOrigin() const;

  void OnMousePressed(const QInputEvent* event) override;
  void onMouseRelease(const QInputEvent* event) override;
  void update();
  void onMouseMove(const QInputEvent* event) override;
  void onWheelEvent(const QInputEvent* event) override;

  void onKeyPressed(const QInputEvent* event) override;
  void onKeyReleased(const QInputEvent* event) override;

protected:
  void moveToLeft();
  void moveToRight();
  void moveToTop();
  void moveToBottom();

  void updateOrientation();
  void updateCameraDirection();
  void updateSpaceImageInformation();

private:
  glm::vec3 mOrigin;
  glm::vec3 mUp;
  glm::vec3 mRight;
  glm::vec3 mForward;
  // [w = cos(theta/2), x = v_x * sin(theta/2), y = v_y * sin(theta/2), z = v_z * sin(theta/2)]
  glm::quat mOrientation;
  glm::vec3 mRotateCenter;
  float mSpeed;
  // 物体绕y轴旋转的弧度
  float mYaw;
  // 物体绕x轴旋转的弧度
  float mPitch;
  // 物体绕z轴旋转的弧度
  float mRoll;
  // 垂直视场角
  float mVFOV;
  // 在使用垂直视场角计算完view plan的高度后通过乘上aspect得到宽度, aspect = width/height
  float mAspect;

  glm::quat mTransform;
  SpaceImageInfo mSpaceImageInfo;

  // 鼠标是否被按下移动
  bool mLeftButtonPressed;
  glm::vec2 mCurrentMousePos;

  // 焦距
  float mFocalLength;
  float mRotateLength;
};
#endif