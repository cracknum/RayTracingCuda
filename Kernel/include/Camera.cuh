#ifndef CAMERA_CUDA_H
#define CAMERA_CUDA_H
#include <glm/glm.hpp>
#include <glm/detail/type_quat.hpp>

class Camera
{
    public:
    // 引入平移，旋转（四元数）
    Camera(const glm::vec3& lookAt, const glm::vec3& lookFrom, const glm::vec3& up, float vfov, float aspect);

    glm::vec3 mLookAt;
    glm::vec3 mLookFrom;
    glm::vec3 mUp;
    glm::vec3 mOrigin;
    
    // 垂直视场角
    float mVFOV;
    // 在使用垂直视场角计算完view plan的高度后通过乘上aspect得到宽度, aspect = width/height
    float mAspect;

    glm::quat mTransform;
};
#endif