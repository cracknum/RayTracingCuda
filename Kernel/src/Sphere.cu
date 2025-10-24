#include "Sphere.cuh"
__device__ Sphere::Sphere() {}
__device__ Sphere::Sphere(const glm::vec3& center, float radius)
  : mCenter(center)
  , mRadius(radius)
{
}

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const
{
  glm::vec3 oc = r.origin() - mCenter;
    float a = glm::dot(r.direction(), r.direction());
    float b = glm::dot(oc, r.direction());
    float c = glm::dot(oc, oc) - mRadius*mRadius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < tMax && temp > tMin) {
            record.t = temp;
            record.point = r.pointAtParameter(record.t);
            record.normal = (record.point - mCenter) / mRadius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < tMax && temp > tMin) {
            record.t = temp;
            record.point = r.pointAtParameter(record.t);
            record.normal = (record.point - mCenter) / mRadius;
            return true;
        }
    }
    return false;
}
