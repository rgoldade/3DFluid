#ifndef LIBRARY_TEST_VELOCITY_FIELD_H
#define LIBRARY_TEST_VELOCITY_FIELD_H

#include "Utilities.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

namespace FluidSim3D::SimTools
{
using namespace Utilities;

class DeformationField
{
public:
    DeformationField() : mySimTime(0), myDeformationPeriod(3) {}

    DeformationField(float startTime, float deformationPeriod)
        : mySimTime(startTime), myDeformationPeriod(deformationPeriod)
    {
    }

    void advanceField(float dt) { mySimTime += dt; }

    Vec3f operator()(float dt, const Vec3f& samplePoint) const
    {
        Vec3f velocity;
        velocity[0] = 2 * sqr(sin(PI * samplePoint[0])) * sin(2 * PI * samplePoint[1]) * sin(2 * PI * samplePoint[2]);
        velocity[1] = -sin(2 * PI * samplePoint[0]) * sqr(sin(PI * samplePoint[1])) * sin(2 * PI * samplePoint[2]);
        velocity[2] = -sin(2 * PI * samplePoint[0]) * sin(2 * PI * samplePoint[1]) * sqr(sin(PI * samplePoint[2]));

        velocity *= cos(PI * (mySimTime + dt) / myDeformationPeriod);
        return velocity;
    }

private:
    float mySimTime, myDeformationPeriod;
};

class CircularField
{
public:
    CircularField(const Vec3f& center, float scale, Axis rotationAxis)
        : myCenter(center), myScale(scale), myRotationAxis(rotationAxis)
    {
    }

    Vec3f operator()(float, const Vec3f& samplePoint) const
    {
        Vec3f velocity(0);
        if (myRotationAxis == Axis::XAXIS)
        {
            velocity[1] = samplePoint[2] - myCenter[2];
            velocity[2] = -(samplePoint[1] - myCenter[1]);
        }
        else if (myRotationAxis == Axis::YAXIS)
        {
            velocity[0] = -(samplePoint[2] - myCenter[2]);
            velocity[2] = samplePoint[0] - myCenter[0];
        }
        else
        {
            assert(myRotationAxis == Axis::ZAXIS);

            velocity[0] = samplePoint[1] - myCenter[1];
            velocity[1] = -(samplePoint[0] - myCenter[0]);
        }

        velocity *= myScale;

        return velocity;
    }

private:
    const Vec3f myCenter;
    const float myScale;
    const Axis myRotationAxis;
};

class NotchedDiskField
{
public:

    // Procedural velocity field
    Vec3f operator()(float, const Vec3f& pos) const
    {
	return Vec3f((PI / 314.) * (50.0 - pos[1]), (PI / 314.) * (pos[0] - 50.0), 0);
    }
};

}  // namespace FluidSim3D::SimTools
#endif