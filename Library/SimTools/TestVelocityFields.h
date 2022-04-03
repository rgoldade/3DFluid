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
//
//class CurlNoise
//{
//public:
//
//    CurlNoise()
//        : myDx(1E-4)
//        , myNoiseScale(1.5)
//        , myNoiseGain(1.3)
//    {}
//
//    Vec3d operator()(double, const Vec3d& pos) const
//    {
//        Vec3d vel;
//        vel[0] = ((potential(pos[0], pos[1] + myDx, pos[2])[2] - potential(pos[0], pos[1] - myDx, pos[2])[2])
//            - (potential(pos[0], pos[1], pos[2] + myDx)[1] - potential(pos[0], pos[1], pos[2] - myDx)[1]))
//            / (2 * myDx);
//        vel[1] = ((potential(pos[0], pos[1], pos[2] + myDx)[0] - potential(pos[0], pos[1], pos[2] - myDx)[0])
//            - (potential(pos[0] + myDx, pos[1], pos[2])[2] - potential(pos[0] - myDx, pos[1], pos[2])[2]))
//            / (2 * myDx);
//        vel[2] = ((potential(pos[0] + myDx, pos[1], pos[2])[1] - potential(pos[0] - myDx, pos[1], pos[2])[1])
//            - (potential(pos[0], pos[1] + myDx, pos[2])[0] - potential(pos[0], pos[1] - myDx, pos[2])[0]))
//            / (2 * myDx);
//
//        return vel;
//    }
//
//private:
//    // Take the curl of this function to get velocity
//    Vec3d potential(double x, double y, double z) const
//    {
//        Vec3d psi = Vec3d::Zero();
//        constexpr double heightFactor = 0.5;
//
//        const Vec3d centre(0.0, 1.0, 0.0);
//        constexpr double radius = 4.0;
//
//        double sx = x / myNoiseScale;
//        double sy = y / myNoiseScale;
//        double sz = z / myNoiseScale;
//
//        Vec3d psi_i(0., 0., noise2(sx, sy, sz));
//
//        double dist = (Vec3d(x, y, z) - centre).norm();
//        double scale = std::max((radius - dist) / radius, 0.0);
//        psi_i *= scale;
//
//        psi += heightFactor * myNoiseGain * psi_i;
//
//        return psi;
//    }
//
//    double noise2(double x, double y, double z) const { return myNoise(Vec3d(z - 203.994, x + 169.47, y - 205.31)); }
//
//    struct Noise
//    {
//        Noise(size_t seed = 171717);
//
//        void reinitialize(size_t seed);
//        double operator()(double x, double y, double z) const;
//        double operator()(const Vec3d& x) const { return (*this)(x[0], x[1], x[2]); }
//
//    protected:
//        static const unsigned int n = 128;
//        Vec3d basis[n];
//        int perm[n];
//
//        size_t hash_index(int i, int j, int k) const
//        {
//            return (size_t)perm[(perm[(perm[i % n] + j) % n] + k) % n];
//        }
//    };
//
//    struct FlowNoise : public Noise
//    {
//        FlowNoise(size_t seed = 171717, double spin_variation = 0.2);
//        void set_time(double t); // period of repetition is approximately 1
//
//    protected:
//        Vec3d original_basis[n];
//        double spin_rate[n];
//        Vec3d spin_axis[n];
//    };
//
//    FlowNoise myNoise;
//
//    double myDx;
//    double myNoiseScale;
//    double myNoiseGain;
//};

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