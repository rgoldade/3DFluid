#ifndef EULERIAN_LIQUID_SIMULATOR_H
#define EULERIAN_LIQUID_SIMULATOR_H

#include "FieldAdvector.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// EulerianLiquidSimulator.h/cpp
// Ryan Goldade 2017
//
// Wrapper class around the staggered MAC grid fluid simulator
// (which stores face-aligned velocities and pressure).
// Handles velocity, surface, viscosity field advection,
// pressure projection, viscosity and velocity extrapolation.
//
////////////////////////////////////

using namespace FluidSim3D;

class EulerianLiquidSimulator
{
public:
    EulerianLiquidSimulator(const Transform& xform, Vec3i size, double cfl = 5)
        : myXform(xform), myDoSolveViscosity(false), myCFL(cfl)
    {
        myLiquidVelocity = VectorGrid<double>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
        mySolidVelocity = VectorGrid<double>(myXform, size, Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

        myLiquidSurface = LevelSet(myXform, size, myCFL);
        mySolidSurface = LevelSet(myXform, size, myCFL);

        myOldPressure = ScalarGrid<double>(myXform, size, 0);
    }

    void setSolidSurface(const LevelSet& solidSurface);
    void setSolidVelocity(const VectorGrid<double>& solidVelocity);
    void setLiquidSurface(const LevelSet& liquidSurface);
    void setLiquidVelocity(const VectorGrid<double>& liquidVelocity);

    void setViscosity(const ScalarGrid<double>& viscosityGrid)
    {
        assert(myLiquidSurface.isGridMatched(viscosityGrid));
        myViscosity = viscosityGrid;
        myDoSolveViscosity = true;
    }

    void setViscosity(double constantViscosity = 1.)
    {
        myViscosity = ScalarGrid<double>(myLiquidSurface.xform(), myLiquidSurface.size(), constantViscosity);
        myDoSolveViscosity = true;
    }

    void unionLiquidSurface(const LevelSet& addedLiquidSurface);

    template <typename ForceSampler>
    void addForce(double dt, const ForceSampler& force);

    void addForce(double dt, const Vec3d& force);

    void advectOldPressure(const double dt);
    void advectLiquidSurface(double dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
    void advectViscosity(double dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
    void advectLiquidVelocity(double dt, IntegrationOrder integrator = IntegrationOrder::RK3);

    // Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
    void runTimestep(double dt);

    // Useful for CFL
    double maxVelocityMagnitude() { return myLiquidVelocity.maxMagnitude(); }

    // Rendering tools
    void drawGrid(Renderer& renderer, bool onlyDrawNarrowBand = false) const;

    void drawLiquidSurface(Renderer& renderer) const;
    void drawLiquidVelocity(Renderer& renderer, Axis planeAxis, double planePosition, double length) const;

    void drawSolidSurface(Renderer& renderer) const;
    void drawSolidVelocity(Renderer& renderer, Axis planeAxis, double planePosition, double length) const;

private:
    // Simulation containers
    VectorGrid<double> myLiquidVelocity, mySolidVelocity;
    LevelSet myLiquidSurface, mySolidSurface;
    ScalarGrid<double> myViscosity;

    Transform myXform;

    bool myDoSolveViscosity;
    double myCFL;

    ScalarGrid<double> myOldPressure;
};

#endif