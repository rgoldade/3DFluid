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
// FieldAdvector.h/cpp
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

template<typename Field, typename VelocityField>
void advectField(float dt, Field& destinationField, const Field& sourceField, const VelocityField& velocity, IntegrationOrder order)
{
	assert(&destinationField != &sourceField);

	tbb::parallel_for(tbb::blocked_range<int>(0, sourceField.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = sourceField.unflatten(cellIndex);

			Vec3f worldPoint = sourceField.indexToWorld(Vec3f(cell));
			worldPoint = Integrator(-dt, worldPoint, velocity, order);

			destinationField(cell) = sourceField.interp(worldPoint);
		}
	});
}

}
#endif