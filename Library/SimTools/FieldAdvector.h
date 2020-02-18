#ifndef LIBRARY_FIELD_ADVECTOR_H
#define LIBRARY_FIELD_ADVECTOR_H

#include "tbb/tbb.h"

#include "Integrator.h"
#include "ScalarGrid.h"
#include "Utilities.h"
#include "Vec.h"
#include "VectorGrid.h"

///////////////////////////////////
//
// AdvectField.h/cpp
// Ryan Goldade 2017
//
// A versatile advection class to handle
// forward advection and semi-Lagrangian
// backtracing.
//
////////////////////////////////////

namespace FluidSim3D::SimTools
{
using namespace Utilities;

template<typename Field>
class FieldAvector
{
public:
	FieldAvector(const Field& source)
		: myField(source)
		{}

	template<typename VelocityField>
	void advectField(float dt, Field& field, const VelocityField& velocity, IntegrationOrder order);

	// Operator overload to sample the velocity field while taking into account
	// solid objects
	Vec3f operator()(float dt, const Vec3f& worldPoint) const;

private:

	const Field& myField;
};

template<typename Field>
template<typename VelocityField>
void FieldAvector<Field>::advectField(float dt, Field& field, const VelocityField& vel, IntegrationOrder order)
{
	assert(&field != &myField);
	assert(myField.isGridMatched(field));

	tbb::parallel_for(tbb::blocked_range<int>(0, myField.voxelCount(), tbbLightGrainSize), [&](tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = field.unflatten(cellIndex);

			Vec3f worldPoint = field.indexToWorld(Vec3f(cell));
			worldPoint = Integrator(-dt, worldPoint, vel, order);

			field(cell) = myField.interp(worldPoint);
		}
	});
}

}
#endif