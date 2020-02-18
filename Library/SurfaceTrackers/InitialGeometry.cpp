#include "InitialGeometry.h"

#include "LevelSet.h"

namespace FluidSim3D::SurfaceTrackers
{

TriMesh makeDiamondMesh(const Vec3f& center, float scale)
{
	std::vector<Vec3f> vertices(6);

	vertices[0] = Vec3f( 0., 1., 0.);
	vertices[1] = Vec3f( 0.,-1., 0.);
	vertices[2] = Vec3f( 1., 0., 0.);

	vertices[3] = Vec3f(-1., 0., 0.);
	vertices[4] = Vec3f( 0., 0., 1.);
	vertices[5] = Vec3f( 0., 0.,-1.);

	for (auto& vertex : vertices) { vertex *= scale; }
	for (auto& vertex : vertices) { vertex += center; }

	std::vector<Vec3i> triFaces(8);

	triFaces[0] = Vec3i(2, 0, 4);
	triFaces[1] = Vec3i(4, 0, 3);
	triFaces[2] = Vec3i(3, 0, 5);
	triFaces[3] = Vec3i(5, 0, 2);
	triFaces[4] = Vec3i(2, 1, 5);
	triFaces[5] = Vec3i(5, 1, 3);
	triFaces[6] = Vec3i(3, 1, 4);
	triFaces[7] = Vec3i(4, 1, 2);

	return TriMesh(triFaces, vertices);
}

TriMesh makeCubeMesh(const Vec3f& center, const Vec3f& scale)
{
	std::vector<Vec3f> vertices(8);

	vertices[0] = Vec3f(-1.,-1., 1.);
	vertices[1] = Vec3f( 1.,-1., 1.);
	vertices[2] = Vec3f(-1., 1., 1.);
	vertices[3] = Vec3f( 1., 1., 1.);

	vertices[4] = Vec3f(-1., 1., -1.);
	vertices[5] = Vec3f( 1., 1., -1.);
	vertices[6] = Vec3f(-1.,-1., -1.);
	vertices[7] = Vec3f( 1.,-1., -1.);

	for (auto& vertex : vertices) { vertex *= scale; }
	for (auto& vertex : vertices) { vertex += center; }

	std::vector<Vec3i> triFaces(12);

	triFaces[0] = Vec3i(0, 1, 2);
	triFaces[1] = Vec3i(2, 1, 3);
	triFaces[2] = Vec3i(2, 3, 4);
	triFaces[3] = Vec3i(4, 3, 5);

	triFaces[4] = Vec3i(4, 5, 6);
	triFaces[5] = Vec3i(6, 5, 7);
	triFaces[6] = Vec3i(6, 7, 0);
	triFaces[7] = Vec3i(0, 7, 1);

	triFaces[8] = Vec3i(1, 7, 3);
	triFaces[9] = Vec3i(3, 7, 5);
	triFaces[10] = Vec3i(6, 0, 4);
	triFaces[11] = Vec3i(4, 0, 2);

	return TriMesh(triFaces, vertices);
}

TriMesh makeSphereMesh(const Vec3f& center, float radius, float dx)
{
	// Build a temporary level set and generate mesh from implicit surface

	Transform xform(dx, center - Vec3f(radius + 5. * dx));
	Vec3i gridSize(2. * radius / dx + 10);

	LevelSet sphereSDF(xform, gridSize);

	tbb::parallel_for(tbb::blocked_range<int>(0, sphereSDF.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i cell = sphereSDF.unflatten(cellIndex);
			Vec3f worldPoint = sphereSDF.indexToWorld(Vec3f(cell));
			float phi = sqrt(sqr(worldPoint[0] - center[0]) + sqr(worldPoint[1] - center[1]) + sqr(worldPoint[2] - center[2])) - radius;
			sphereSDF(cell) = phi;
		}
	});

	return sphereSDF.buildMesh();
}

}