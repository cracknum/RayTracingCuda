#include "Sphere.cuh"

#include "Material.cuh"
#include "HitRecord.cuh"
__device__ Sphere::Sphere() {}

__device__
Sphere::~Sphere() {
  if (mMaterial)
  {
    delete mMaterial;
  }
  
}
__device__ Sphere::Sphere(const glm::vec3& center, float radius, Material* material)
  : Sphere(center, center, radius, material)
{
}

__device__ Sphere::Sphere(
  const glm::vec3& startCenter, const glm::vec3& endCenter, float radius, Material* material)
  :mCenter(startCenter, endCenter - startCenter), mRadius(fmaxf(0.0f, radius)), mMaterial(material)
{

}

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& record) const
{
  glm::vec3 currentCenter = mCenter.pointAtParameter(r.renderTime());
  glm::vec3 oc = r.origin() - currentCenter;
  float a = glm::dot(r.direction(), r.direction());
  float b = glm::dot(oc, r.direction());
  float c = glm::dot(oc, oc) - mRadius * mRadius;
  float discriminant = b * b - a * c;
  if (discriminant > 0)
  {
    float temp = (-b - sqrt(discriminant)) / a;
    if (temp < tMax && temp > tMin)
    {
      record.t = temp;
      record.point = r.pointAtParameter(record.t);
      record.material = mMaterial;
      glm::vec3 outwardNormal = (record.point - currentCenter) / mRadius;
      record.setFaceNormal(r, outwardNormal);
      return true;
    }
    temp = (-b + sqrt(discriminant)) / a;
    if (temp < tMax && temp > tMin)
    {
      record.t = temp;
      record.point = r.pointAtParameter(record.t);
      glm::vec3 outwardNormal = (record.point - currentCenter) / mRadius;
      record.setFaceNormal(r, outwardNormal);
      record.material = mMaterial;
      return true;
    }
  }
  return false;
}
