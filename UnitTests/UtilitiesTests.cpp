#include "gtest/gtest.h"

#include "tbb/blocked_range.h"
#include "tbb/enumerable_thread_specific.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_sort.h"

#include "Utilities.h"

using namespace FluidSim3D;

TEST(UTILITIES_TESTS, MERGE_VECTOR_TEST)
{
	tbb::enumerable_thread_specific<VecVec3i> parallelVectors;

	int testSize = 10000;
	tbb::parallel_for(tbb::blocked_range<int>(0, testSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localVectors = parallelVectors.local();
		for (int index = range.begin(); index != range.end(); ++index)
		{
			localVectors.emplace_back(index, 2 * index, 3 * index);
		}
	});

	VecVec3i combinedVectors;
	mergeLocalThreadVectors(combinedVectors, parallelVectors);

	ASSERT_EQ(combinedVectors.size(), testSize);

	tbb::parallel_sort(combinedVectors.begin(), combinedVectors.end(), [](const Vec3i& vec0, const Vec3i& vec1)
	{
		return std::tie(vec0[0], vec0[1], vec0[2]) < std::tie(vec1[0], vec1[1], vec1[2]);
	});

	for (int index = 0; index != combinedVectors.size(); ++index)
	{
		const Vec3i& vec = combinedVectors[index];
		EXPECT_EQ(vec[0], index);
		EXPECT_EQ(vec[1], 2 * index);
		EXPECT_EQ(vec[2], 3 * index);
	}

	parallelVectors.clear();
	tbb::parallel_for(tbb::blocked_range<int>(0, testSize), [&](const tbb::blocked_range<int>& range)
	{
		auto& localVectors = parallelVectors.local();
		for (int index = range.begin(); index != range.end(); ++index)
		{
			int offsetIndex = index + testSize;
			localVectors.emplace_back(offsetIndex, 2 * offsetIndex, 3 * offsetIndex);
		}
	});

	mergeLocalThreadVectors(combinedVectors, parallelVectors);

	ASSERT_EQ(combinedVectors.size(), 2 * testSize);

	tbb::parallel_sort(combinedVectors.begin(), combinedVectors.end(), [](const Vec3i& vec0, const Vec3i& vec1)
	{
		return std::tie(vec0[0], vec0[1], vec0[2]) < std::tie(vec1[0], vec1[1], vec1[2]);
	});

	for (int index = 0; index != combinedVectors.size(); ++index)
	{
		const Vec3i& vec = combinedVectors[index];
		EXPECT_EQ(vec[0], index);
		EXPECT_EQ(vec[1], 2 * index);
		EXPECT_EQ(vec[2], 3 * index);
	}
}

TEST(UTILITIES_TESTS, BARYCENTER_TEST)
{
	int testSize = 1000;
	for (int index = 0; index != testSize; ++index)
	{
		Vec3d v0 = 1000. * Vec3d::Random();
		Vec3d v1 = v0 + Vec3d::Random();
		Vec3d v2 = v1 + Vec3d::Random();

		// Check first vertex weight
		{
			Vec3d vb = v0;

			Vec3d baryweights = computeBarycenters(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual(baryweights[0], 1., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[1], 0., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[2], 0., 1e-5, false));
		}

		// Check second vertex weight
		{
			Vec3d vb = v1;

			Vec3d baryweights = computeBarycenters(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual(baryweights[0], 0., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[1], 1., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[2], 0., 1e-5, false));
		}

		// Check third vertex weight
		{
			Vec3d vb = v2;

			Vec3d baryweights = computeBarycenters(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual(baryweights[0], 0., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[1], 0., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[2], 1., 1e-5, false));
		}

		// Check midpoint weight
		{
			Vec3d vb = (v0 + v1 + v2) / 3.;

			Vec3d baryweights = computeBarycenters(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual(baryweights[0], 1. / 3., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[1], 1. / 3., 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(baryweights[2], 1. / 3., 1e-5, false));
		}
	}
}

TEST(UTILITIES_TESTS, BASIC_POINT_TO_TRIANGLE_PROJECTION_TEST)
{
	Vec3d v0(0., 0., 0.);
	Vec3d v1(1., 0., 0.);
	Vec3d v2(1., 1., 1.);

	Vec3d midpoint = (v0 + v1 + v2) / 3.;

	// Test projectionion to first vertex
	{
		Vec3d vb = v0 + .1 * (v0 - midpoint).normalized();
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, v0));
	}

	// Test projection to second vertex
	{
		Vec3d vb = v1 + .1 * (v1 - midpoint).normalized();
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, v1));
	}

	// Test projection to third vertex
	{
		Vec3d vb = v2 + .1 * (v2 - midpoint).normalized();
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, v2));
	}

	Vec3d tangent0 = v1 - v0;
	Vec3d tangent1 = v2 - v0;

	Vec3d triNormal = tangent0.cross(tangent1).normalized();

	// Test project to first edge
	{
		Vec3d edgeMidpoint = (v0 + v1) / 2.;
		Vec3d vb = edgeMidpoint + .1 * triNormal;
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, edgeMidpoint));
	}

	// Test project to second edge
	{
		Vec3d edgeMidpoint = (v1 + v2) / 2.;
		Vec3d vb = edgeMidpoint + .1 * triNormal;
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, edgeMidpoint));
	}

	// Test project to third edge
	{
		Vec3d edgeMidpoint = (v2 + v0) / 2.;
		Vec3d vb = edgeMidpoint + .1 * triNormal;
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, edgeMidpoint));
	}

	// Test project to midpoint
	{
		Vec3d vb = midpoint + .1 * triNormal;
		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);
		EXPECT_TRUE(isMatrixNearlyEqual(vp, midpoint));
	}
}


TEST(UTILITIES_TESTS, POINT_TO_TRIANGLE_PROJECTION_TEST)
{
	int testSize = 1000;
	for (int index = 0; index != testSize; ++index)
	{
		Vec3d v0 = Vec3d::Random();
		Vec3d v1 = Vec3d::Random();
		Vec3d v2 = Vec3d::Random();

		Vec3d tangent0 = v1 - v0;
		Vec3d tangent1 = v2 - v0;

		Vec3d triNormal = tangent0.cross(tangent1).normalized();

		Vec3d midpoint = (v0 + v1 + v2) / 3.;

		//
		// Normal offsets
		//

		// Offset from first vertex
		{
			Vec3d vb = v0 + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v0).norm(), 0., 1e-5, false));

			vb = v0 - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v0).norm(), 0., 1e-5, false));
		}

		// Offset from second vertex
		{
			Vec3d vb = v1 + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v1).norm(), 0., 1e-5, false));

			vb = v1 - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v1).norm(), 0., 1e-5, false));
		}

		// Offset from third vertex
		{
			Vec3d vb = v2 + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v2).norm(), 0., 1e-5, false));

			vb = v2 - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - v2).norm(), 0., 1e-5, false));
		}


		// Offset from first edge
		{
			Vec3d edgeMidpoint = (v0 + v1) / 2.;
			Vec3d vb = edgeMidpoint + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));

			vb = edgeMidpoint - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));
		}

		// Offset from second edge
		{
			Vec3d edgeMidpoint = (v1 + v2) / 2.;
			Vec3d vb = edgeMidpoint + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));

			vb = edgeMidpoint - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));
		}

		// Offset from third vertex
		{
			Vec3d edgeMidpoint = (v2 + v0) / 2.;
			Vec3d vb = edgeMidpoint + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));

			vb = edgeMidpoint - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - edgeMidpoint).norm(), 0., 1e-5, false));
		}

		// Offset from midpoint vertex
		{
			Vec3d vb = midpoint + .1 * triNormal;
			Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - midpoint).norm(), 0., 1e-5, false));

			vb = midpoint - .1 * triNormal;
			vp = pointToTriangleProjection(vb, v0, v1, v2);

			EXPECT_TRUE(isNearlyEqual((vp - midpoint).norm(), 0., 1e-5, false));
		}
	}
}

TEST(UTILITIES_TESTS, POINT_TO_TRIANGLE_DISTANCE_BOUNDS_TEST)
{
	int testSize = 1000;
	for (int index = 0; index != testSize; ++index)
	{
		Vec3d v0 = Vec3d::Random();
		Vec3d v1 = Vec3d::Random();
		Vec3d v2 = Vec3d::Random();

		Vec3d vb = Vec3d::Random();

		Vec3d vp = pointToTriangleProjection(vb, v0, v1, v2);

		double minVertexDist = std::min((vb - v0).norm(), std::min((vb - v1).norm(), (vb - v2).norm()));

		EXPECT_LE((vb - vp).norm(), minVertexDist);
	}
}