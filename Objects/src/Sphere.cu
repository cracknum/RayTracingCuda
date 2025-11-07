#include "Sphere.cuh"

#include "HitRecord.cuh"
#include "Material.cuh"
__device__ Sphere::Sphere() {}

__device__ Sphere::~Sphere()
{
  if (mMaterial)
  {
    delete mMaterial;
  }
}
__device__ Sphere::Sphere(const glm::vec3& center, float radius, Material* material)
  : Sphere(center, center, radius, material)
{
  updateBoundingBox(center, radius);
}

__device__
void Sphere::updateBoundingBox(const glm::vec3& center, float radius)
{
  glm::vec3 corner[2] = { center - radius, center + radius };
  mBoundingBox = AABB(Interval<float>(corner[0].x, corner[1].x),
    Interval<float>(corner[0].y, corner[1].y), Interval<float>(corner[0].z, corner[1].z));
}

__device__ Sphere::Sphere(
  const glm::vec3& startCenter, const glm::vec3& endCenter, float radius, Material* material)
  : mCenter(startCenter, endCenter - startCenter)
  , mRadius(fmaxf(0.0f, radius))
  , mMaterial(material)
{
   updateBoundingBox(startCenter, radius);
}

__device__ bool Sphere::hit(const Ray& r, float tMin, float tMax, HitRecord& record)
{
  glm::vec3 currentCenter = mCenter.pointAtParameter(r.renderTime());
  updateBoundingBox(currentCenter, mRadius);

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