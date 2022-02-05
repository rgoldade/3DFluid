#ifndef FLUIDSIM3D_TEST_VELOCITY_FIELD_H
#define FLUIDSIM3D_TEST_VELOCITY_FIELD_H

#include "Utilities.h"

///////////////////////////////////
//
// TestVelocityFields.h
// Ryan Goldade 2018
//
////////////////////////////////////

namespace FluidSim3D
{

class DeformationField
{
public:
    DeformationField() : mySimTime(0), myDeformationPeriod(3) {}

    DeformationField(double startTime, double deformationPeriod)
        : mySimTime(startTime), myDeformationPeriod(deformationPeriod)
    {}

    void advanceField(double dt) { mySimTime += dt; }

    Vec3d operator()(double dt, const Vec3d& samplePoint) const
    {
        Vec3d velocity;
        velocity[0] = 2 * std::pow(std::sin(PI * samplePoint[0]), 2) * std::sin(2 * PI * samplePoint[1]) * std::sin(2 * PI * samplePoint[2]);
        velocity[1] = -std::sin(2 * PI * samplePoint[0]) * std::pow(std::sin(PI * samplePoint[1]), 2) * std::sin(2 * PI * samplePoint[2]);
        velocity[2] = -std::sin(2 * PI * samplePoint[0]) * std::sin(2 * PI * samplePoint[1]) * std::pow(std::sin(PI * samplePoint[2]), 2);

        velocity *= std::cos(PI * (mySimTime + dt) / myDeformationPeriod);
        return velocity;
    }

private:
    double mySimTime, myDeformationPeriod;
};

class CircularField
{
public:
    CircularField(const Vec3d& center, double scale, Axis rotationAxis)
        : myCenter(center), myScale(scale), myRotationAxis(rotationAxis)
    {}

    Vec3d operator()(double, const Vec3d& samplePoint) const
    {
        Vec3d velocity = Vec3d::Zero();
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
    const Vec3d myCenter;
    const double myScale;
    const Axis myRotationAxis;
};

class NotchedDiskField
{
public:

    // Procedural velocity field
    Vec3d operator()(float, const Vec3d& pos) const
    {
	return Vec3d((PI / 314.) * (50.0 - pos[1]), (PI / 314.) * (pos[0] - 50.0), 0);
    }
};

}

#endif