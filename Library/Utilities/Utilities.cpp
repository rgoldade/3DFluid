#include "Utilities.h"

namespace FluidSim3D
{

Vec3d computeBarycenters(const Vec3d& vb, const Vec3d& v0, const Vec3d& v1, const Vec3d& v2)
{
    Vec3d d1 = v1 - v0;
    Vec3d d2 = v2 - v0;
    Vec3d db = vb - v0;

    Matrix2x2d AtA;
    AtA(0, 0) = d1.dot(d1);
    AtA(0, 1) = d1.dot(d2);
    AtA(1, 0) = AtA(0, 1);
    AtA(1, 1) = d2.dot(d2);

    Vec2d Atb;
    Atb(0) = d1.dot(db);
    Atb(1) = d2.dot(db);

    Vec3d barycenter;
    Matrix2x2d invA = AtA.inverse();
    barycenter.segment<2>(1) = invA * Atb;

    // Error correction clean up.
    // Note that if x = x^n + e^n, then residual r^n = b - A x^n = b - A (x - e^n) = A e^n.
    // Therefore x^(n+1) = x^n + A^-1 r^n approximates the true solution.
    barycenter.segment<2>(1) += invA * (Atb - AtA * barycenter.segment<2>(1));
    barycenter[0] = 1.0 - barycenter[1] - barycenter[2];

    return barycenter;
}

// Helper function to project a point to a triangle in 3-D
// Taken from libIGL -- https://github.com/libigl/libigl/blob/main/include/igl/point_simplex_squared_distance.cpp
Vec3d pointToTriangleProjection(const Vec3d& vp, const Vec3d& v0, const Vec3d& v1, const Vec3d& v2)
{
    // Check if P in vertex region outside A
    Vec3d v10 = v1 - v0;
    Vec3d v20 = v2 - v0;
    Vec3d vp0 = vp - v0;
    double d1 = v10.dot(vp0);
    double d2 = v20.dot(vp0);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return v0;
    }
    // Check if P in vertex region outside B
    Vec3d bp = vp - v1;
    double d3 = v10.dot(bp);
    double d4 = v20.dot(bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        return v1;
    }
    // Check if P in edge region of AB, if so return projection of P onto AB
    double vc = d1 * d4 - d3 * d2;
    if (v0 != v1)
    {
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) 
        {
            double v = d1 / (d1 - d3);
            return v0 + v * v10;
        }
    }
    // Check if P in vertex region outside C
    Vec3d vp2 = vp - v2;
    double d5 = v10.dot(vp2);
    double d6 = v20.dot(vp2);
    if (d6 >= 0.0 && d5 <= d6)
    {
        return v2;
    }
    // Check if P in edge region of AC, if so return projection of P onto AC
    double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) 
    {
        double w = d2 / (d2 - d6);
        return v0 + w * v20;
    }
    // Check if P in edge region of BC, if so return projection of P onto BC
    double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + w * (v2 - v1);
    }
    // P inside face region. Compute Q through its barycentric coordinates (u,v,w)
    double denom = 1.0 / (va + vb + vc);
    double v = vb * denom;
    double w = vc * denom;
    return v0 + v10 * v + v20 * w; // = u*a + v*b + w*c, u = va * denom = 1.0-v-w
};

double computeRayBBoxIntersection(const AlignedBox3d& bbox, const Vec3d& rayOrigin, const Vec3d& rayDirection)
{
    AlignedBox3d expandedBbox(bbox);
    expandedBbox.extend(bbox.max() + Vec3d::Constant(1e-3));
    expandedBbox.extend(bbox.min() - Vec3d::Constant(1e-3));

    double alpha = std::numeric_limits<double>::max();
    for (int axis : {0, 1, 2})
        for (int direction : {0, 1})
        {
            double planeValue;
            if (direction == 0)
            {
                planeValue = bbox.min()[axis];
            }
            else
            {
                planeValue = bbox.max()[axis];
            }

            double localAlpha = (planeValue - rayOrigin[axis]) / rayDirection[axis];
            Vec3d point = rayOrigin + localAlpha * rayDirection;

            if (expandedBbox.contains(point))
            {
                alpha = std::min(alpha, localAlpha);
            }
        }

    return alpha;
}

}
