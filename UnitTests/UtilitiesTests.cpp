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

TEST(UTILITIES_TESTS, LERP_TEST)
{
	EXPECT_DOUBLE_EQ(lerp(0., 1., 0.), 0.);
	EXPECT_DOUBLE_EQ(lerp(0., 1., 1.), 1.);
	EXPECT_DOUBLE_EQ(lerp(0., 1., 0.5), 0.5);
	EXPECT_DOUBLE_EQ(lerp(2., 4., 0.25), 2.5);
}

TEST(UTILITIES_TESTS, TRILERP_CONVERGENCE_TEST)
{
	auto testFunc = [](const Vec3d& p) -> double
	{
		return std::sin(PI * p[0]) * std::sin(PI * p[1]) * std::sin(PI * p[2]);
	};

	int baseN = 8;
	int testSize = 4;
	std::vector<double> errors;

	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		int N = baseN * int(std::pow(2, testIndex));
		double h = 1. / double(N);

		double maxError = 0;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				for (int k = 0; k < N; ++k)
				{
					Vec3d p = (Vec3d(i, j, k) + Vec3d::Constant(0.5)) * h;

					double v000 = testFunc(Vec3d(i, j, k) * h);
					double v100 = testFunc(Vec3d(i + 1, j, k) * h);
					double v010 = testFunc(Vec3d(i, j + 1, k) * h);
					double v110 = testFunc(Vec3d(i + 1, j + 1, k) * h);
					double v001 = testFunc(Vec3d(i, j, k + 1) * h);
					double v101 = testFunc(Vec3d(i + 1, j, k + 1) * h);
					double v011 = testFunc(Vec3d(i, j + 1, k + 1) * h);
					double v111 = testFunc(Vec3d(i + 1, j + 1, k + 1) * h);

					double interp = trilerp(v000, v100, v010, v110, v001, v101, v011, v111, 0.5, 0.5, 0.5);
					double exact = testFunc(p);
					maxError = std::max(maxError, std::fabs(interp - exact));
				}

		errors.push_back(maxError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.5);
	}
}

TEST(UTILITIES_TESTS, TRILERP_GRADIENT_CONVERGENCE_TEST)
{
	auto testFunc = [](const Vec3d& p) -> double
	{
		return std::sin(PI * p[0]) * std::sin(PI * p[1]) * std::sin(PI * p[2]);
	};

	auto testFuncGrad = [](const Vec3d& p) -> Vec3d
	{
		return Vec3d(PI * std::cos(PI * p[0]) * std::sin(PI * p[1]) * std::sin(PI * p[2]),
					 PI * std::sin(PI * p[0]) * std::cos(PI * p[1]) * std::sin(PI * p[2]),
					 PI * std::sin(PI * p[0]) * std::sin(PI * p[1]) * std::cos(PI * p[2]));
	};

	int baseN = 8;
	int testSize = 4;
	std::vector<double> errors;

	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		int N = baseN * int(std::pow(2, testIndex));
		double h = 1. / double(N);

		double maxError = 0;
		for (int i = 0; i < N; ++i)
			for (int j = 0; j < N; ++j)
				for (int k = 0; k < N; ++k)
				{
					Vec3d p = (Vec3d(i, j, k) + Vec3d::Constant(0.5)) * h;

					double v000 = testFunc(Vec3d(i, j, k) * h);
					double v100 = testFunc(Vec3d(i + 1, j, k) * h);
					double v010 = testFunc(Vec3d(i, j + 1, k) * h);
					double v110 = testFunc(Vec3d(i + 1, j + 1, k) * h);
					double v001 = testFunc(Vec3d(i, j, k + 1) * h);
					double v101 = testFunc(Vec3d(i + 1, j, k + 1) * h);
					double v011 = testFunc(Vec3d(i, j + 1, k + 1) * h);
					double v111 = testFunc(Vec3d(i + 1, j + 1, k + 1) * h);

					Vec3d interpGrad = trilerpGradient(v000, v100, v010, v110, v001, v101, v011, v111, 0.5, 0.5, 0.5) / h;
					Vec3d exactGrad = testFuncGrad(p);
					maxError = std::max(maxError, (interpGrad - exactGrad).norm());
				}

		errors.push_back(maxError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 1.8);
	}
}

TEST(UTILITIES_TESTS, CUBIC_INTERP_CONVERGENCE_TEST)
{
	auto testFunc = [](double x) -> double
	{
		return std::sin(PI * x);
	};

	int baseN = 8;
	int testSize = 5;
	std::vector<double> errors;

	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		int N = baseN * int(std::pow(2, testIndex));
		double h = 1. / double(N);

		double maxError = 0;
		for (int i = 1; i < N - 1; ++i)
		{
			for (double fx = 0.1; fx < 1.0; fx += 0.2)
			{
				double v_1 = testFunc((i - 1) * h);
				double v0 = testFunc(i * h);
				double v1 = testFunc((i + 1) * h);
				double v2 = testFunc((i + 2) * h);

				double interp = cubicInterp(v_1, v0, v1, v2, fx);
				double exact = testFunc((i + fx) * h);
				maxError = std::max(maxError, std::fabs(interp - exact));
			}
		}

		errors.push_back(maxError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 8.);
	}
}

TEST(UTILITIES_TESTS, CUBIC_INTERP_GRADIENT_CONVERGENCE_TEST)
{
	auto testFunc = [](double x) -> double
	{
		return std::sin(PI * x);
	};

	auto testFuncGrad = [](double x) -> double
	{
		return PI * std::cos(PI * x);
	};

	int baseN = 8;
	int testSize = 5;
	std::vector<double> errors;

	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		int N = baseN * int(std::pow(2, testIndex));
		double h = 1. / double(N);

		double maxError = 0;
		for (int i = 1; i < N - 1; ++i)
		{
			for (double fx = 0.1; fx < 1.0; fx += 0.2)
			{
				double v_1 = testFunc((i - 1) * h);
				double v0 = testFunc(i * h);
				double v1 = testFunc((i + 1) * h);
				double v2 = testFunc((i + 2) * h);

				double interpGrad = cubicInterpGradient(v_1, v0, v1, v2, fx) / h;
				double exactGrad = testFuncGrad((i + fx) * h);
				maxError = std::max(maxError, std::fabs(interpGrad - exactGrad));
			}
		}

		errors.push_back(maxError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.8);
	}
}