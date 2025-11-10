#include "Quad.cuh"
#include "Material.cuh"
#include "HitRecord.cuh"
#include "Ray.cuh"

__device__ Quad::Quad(
  const glm::vec3& Q, const glm::vec3& u, const glm::vec3& v, Material* material)
  : mQ(Q)
  , mU(u)
  , mV(v)
  , mMaterial(material)
{
    auto n = glm::cross(mU, mV);
    mNormal = glm::normalize(n);
    mW = n / dot(n, n);
    mD = glm::dot(mNormal, Q);
}

__device__ bool Quad::hit(const Ray& r, float tMin, float tMax, HitRecord& record)
{
    auto denom = glm::dot(mNormal, r.direction());
  if (abs(denom) < 1e-6)
  {
    return false;
  }

  auto t = (mD - dot(mNormal, r.origin())) / denom;

  if (tMin > t || tMax < t)
  {
    return false;
  }

  auto intersect = r.pointAtParameter(t);

  auto p = intersect - mQ;
  auto alpha = dot(mW, glm::cross(p, mV));
  auto beta = dot(mW, glm::cross(mU, p));

  if (alpha < 0 || alpha > 1 || beta < 0 || beta > 1)
  {
    return false;
  }

  record.u = alpha;
  record.v = beta;

  record.t = t;
  record.material = mMaterial;
  record.setFaceNormal(r, mNormal);
  record.point = intersect;

  return true;
  
}
