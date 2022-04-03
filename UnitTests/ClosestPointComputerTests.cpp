#include "gtest/gtest.h"

#include "ClosestPointComputer.h"
#include "InitialGeometry.h"

using namespace FluidSim3D;

TEST(CLOSEST_POINT_COMPUTER_TESTS, SELF_RETURN_TRIANGLE)
{
	const Vec3d center = 100. * Vec3d::Random();
	double dx = .01;
	double radius = 1.;
	TriMesh mesh = makeSphereMesh(center, radius, 3);

	// Run closest point for each triangle mid point, check that the closest point found is the midpoint and the face index matches
	ClosestPointComputer cpComputer(mesh);

	for (int triIndex = 0; triIndex != mesh.triangleCount(); ++triIndex)
	{
		const Vec3i& tri = mesh.triangle(triIndex);

		Vec3d midpoint = (mesh.vertex(tri[0]) + mesh.vertex(tri[1]) + mesh.vertex(tri[2])) / 3.;

		auto cpPoint = cpComputer.computeClosestPoint(midpoint, 2 * dx);

		EXPECT_TRUE(isNearlyEqual((cpPoint.first - midpoint).norm(), 0., 1e-5, false));
		EXPECT_EQ(cpPoint.second, triIndex);
	}
}

TEST(CLOSEST_POINT_COMPUTER_TESTS, BRUTE_FORCE_VERIFACTION)
{
	const Vec3d center = 100. * Vec3d::Random();
	double dx = .01;
	double radius = 1.;
	TriMesh mesh = makeSphereMesh(center, radius, 3);

	ClosestPointComputer cpComputer(mesh);

	int testSize = 100;
	for (int testIndex = 0; testIndex != testSize; ++testIndex)
	{
		Vec3d samplePoint = center + 1.5 * radius * Vec3d::Random();

		auto cpPoint = cpComputer.computeClosestPoint(samplePoint);

		// Brute force closest point
		int bruteCPIndex = -1;
		Vec3d bruteCPPoint;
		double bruteCPDist = std::numeric_limits<double>::max();
		for (int triIndex = 0; triIndex != mesh.triangleCount(); ++triIndex)
		{
			const Vec3i& tri = mesh.triangle(triIndex);
			Vec3d vp = pointToTriangleProjection(samplePoint, mesh.vertex(tri[0]), mesh.vertex(tri[1]), mesh.vertex(tri[2]));

			double localDist = (vp - samplePoint).norm();

			if (localDist < bruteCPDist)
			{
				bruteCPPoint = vp;
				bruteCPDist = localDist;
				bruteCPIndex = triIndex;
			}
		}

		EXPECT_TRUE(isNearlyEqual(bruteCPDist, (cpPoint.first - samplePoint).norm()));
	}
}

TEST(CLOSEST_POINT_COMPUTER_TESTS, CENTER_CIRCLE)
{
	const Vec3d center = 100. * Vec3d::Random();
	double dx = .01;
	double radius = 1.;
	TriMesh mesh = makeSphereMesh(center, radius, 6);

	ClosestPointComputer cpComputer(mesh);

	{
		auto cpPoint = cpComputer.computeClosestPoint(center, dx);
		EXPECT_TRUE(cpPoint.second == -1);
	}

	{
		auto cpPoint = cpComputer.computeClosestPoint(center);
		EXPECT_TRUE(isNearlyEqual(radius, (cpPoint.first - center).norm(), 1e-4)) << "Radius: " << radius << ". CP distance: " << (cpPoint.first - center).norm();
	}
}