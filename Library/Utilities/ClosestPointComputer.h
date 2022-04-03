#ifndef FLUIDSIM3D_CLOSEST_POINT_COMPUTER_H
#define FLUIDSIM3D_CLOSEST_POINT_COMPUTER_H

#include "Transform.h"
#include "TriMesh.h"
#include "UniformGrid.h"
#include "Utilities.h"

namespace FluidSim3D
{

class ClosestPointComputer
{
public:
	ClosestPointComputer();

	ClosestPointComputer(const TriMesh& mesh);

	std::pair<Vec3d, int> computeClosestPoint(const Vec3d& queryPoint, double radius = std::numeric_limits<double>::max());

private:

	TriMesh myMesh;

	UniformGrid<std::vector<int>> myMeshGrid;

	Transform myXform;
};

}
#endif
