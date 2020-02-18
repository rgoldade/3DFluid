#ifndef EULERIAN_LIQUID_SIMULATOR_H
#define EULERIAN_LIQUID_SIMULATOR_H

#include "FieldAdvector.h"
#include "Integrator.h"
#include "LevelSet.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "Vec.h"
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

using namespace FluidSim3D::SimTools;
using namespace FluidSim3D::SurfaceTrackers;
using namespace FluidSim3D::Utilities;

class EulerianLiquidSimulator
{
public:
	EulerianLiquidSimulator(const Transform& xform, Vec3i size, float cfl = 5)
		: myXform(xform)
		, myDoSolveViscosity(false)
		, myCFL(cfl)
	{
		myLiquidVelocity = VectorGrid<float>(myXform, size, VectorGridSettings::SampleType::STAGGERED);
		mySolidVelocity = VectorGrid<float>(myXform, size, 0., VectorGridSettings::SampleType::STAGGERED);

		myLiquidSurface = LevelSet(myXform, size, myCFL);
		mySolidSurface = LevelSet(myXform, size, myCFL);

		myOldPressure = ScalarGrid<float>(myXform, size, 0);
	}

	void setSolidSurface(const LevelSet& solidSurface);
	void setSolidVelocity(const VectorGrid<float>& solidVelocity);
	void setLiquidSurface(const LevelSet& liquidSurface);
	void setLiquidVelocity(const VectorGrid<float>& liquidVelocity);

	void setViscosity(const ScalarGrid<float>& viscosityGrid)
	{
		assert(myLiquidSurface.isGridMatched(viscosityGrid));
		myViscosity = viscosityGrid;
		myDoSolveViscosity = true;
	}

	void setViscosity(float constantViscosity = 1.)
	{
		myViscosity = ScalarGrid<float>(myLiquidSurface.xform(), myLiquidSurface.size(), constantViscosity);
		myDoSolveViscosity = true;
	}

	void unionLiquidSurface(const LevelSet& addedLiquidSurface);

	template<typename ForceSampler>
	void addForce(float dt, const ForceSampler& force);

	void addForce(float dt, const Vec3f& force);

	void advectOldPressure(const float dt);
	void advectLiquidSurface(float dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectViscosity(float dt, IntegrationOrder integrator = IntegrationOrder::FORWARDEULER);
	void advectLiquidVelocity(float dt, IntegrationOrder integrator = IntegrationOrder::RK3);

	// Perform pressure project, viscosity solver, extrapolation, surface and velocity advection
	void runTimestep(float dt);

	// Useful for CFL
	float maxVelocityMagnitude() { return myLiquidVelocity.maxMagnitude(); }

	// Rendering tools
	void drawGrid(Renderer& renderer, bool onlyDrawNarrowBand = false) const;
	
	void drawLiquidSurface(Renderer& renderer) const;
	void drawLiquidVelocity(Renderer& renderer, Axis planeAxis, float planePosition, float length) const;

	void drawSolidSurface(Renderer& renderer) const;
	void drawSolidVelocity(Renderer& renderer, Axis planeAxis, float planePosition, float length) const;



private:

	// Simulation containers
	VectorGrid<float> myLiquidVelocity, mySolidVelocity;
	LevelSet myLiquidSurface, mySolidSurface;
	ScalarGrid<float> myViscosity;

	Transform myXform;

	bool myDoSolveViscosity;
	float myCFL;

	ScalarGrid<float> myOldPressure;
};

#endif