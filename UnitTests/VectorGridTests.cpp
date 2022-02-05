#include <random>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "gtest/gtest.h"

#include "GridUtilities.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

static void testSampleType(const VectorGridSettings::SampleType sampleType)
{
	double dx = .01;
	Vec3d origin = Vec3d::Random();
	Vec3i cellSize = Vec3i(10, 20, 30);

	Transform xform(dx, origin);

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	EXPECT_EQ(testGrid.sampleType(), sampleType);

	EXPECT_TRUE(testGrid.gridSize() == cellSize);

	// Test sample count related to the grid size 
	switch (sampleType)
	{
	case VectorGridSettings::SampleType::CENTER:
	{
		EXPECT_TRUE(testGrid.size(0) == cellSize);
		EXPECT_TRUE(testGrid.size(1) == cellSize);
        EXPECT_TRUE(testGrid.size(2) == cellSize);
		break;
	}
	case VectorGridSettings::SampleType::STAGGERED:
	{
		EXPECT_TRUE(testGrid.size(0) == (cellSize + Vec3i(1, 0, 0)).eval());
		EXPECT_TRUE(testGrid.size(1) == (cellSize + Vec3i(0, 1, 0)).eval());
        EXPECT_TRUE(testGrid.size(2) == (cellSize + Vec3i(0, 0, 1)).eval());
		break;
	}
	case VectorGridSettings::SampleType::NODE:
	{
		EXPECT_TRUE(testGrid.size(0) == (cellSize + Vec3i::Ones()).eval());
		EXPECT_TRUE(testGrid.size(1) == (cellSize + Vec3i::Ones()).eval());
        EXPECT_TRUE(testGrid.size(2) == (cellSize + Vec3i::Ones()).eval());
		break;
	}
	case VectorGridSettings::SampleType::EDGE:
	{
		EXPECT_TRUE(testGrid.size(0) == (cellSize + Vec3i(0, 1, 1)).eval());
		EXPECT_TRUE(testGrid.size(1) == (cellSize + Vec3i(1, 0, 1)).eval());
        EXPECT_TRUE(testGrid.size(2) == (cellSize + Vec3i(1, 1, 0)).eval());
	}
	}

	// Test index-to-world and back
	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
        {
            Vec3d indexPoint = testGrid.worldToIndex(testGrid.indexToWorld(coord.cast<double>(), axis), axis);
            EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(indexPoint[2], double(coord[2]), 1e-5, false));
        });
	}

	// Test sampling
	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
		{
			Vec3d worldPoint;
			switch (sampleType)
			{
				case VectorGridSettings::SampleType::CENTER:
				{
					worldPoint = origin + dx * (coord.cast<double>() + .5 * Vec3d::Ones());
					break;
				}
				case VectorGridSettings::SampleType::STAGGERED:
				{
					if (axis == 0)
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., .5, .5));
					else if (axis == 1)
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, 0., .5));
                    else
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, .5, 0.));
					break;
				}
				case VectorGridSettings::SampleType::NODE:
				{
					worldPoint = origin + dx * coord.cast<double>();
					break;
				}
				case VectorGridSettings::SampleType::EDGE:
				{
					if (axis == 0)
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, 0., 0.));
					else if (axis == 1)
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., .5, 0.));
                    else
						worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., 0., .5));
				}				
			}

			Vec3d indexPoint = testGrid.worldToIndex(worldPoint, axis);

			EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
			EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
            EXPECT_TRUE(isNearlyEqual(indexPoint[2], double(coord[2]), 1e-5, false));
		});
	}

    // Assign random values to grid
    for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
		{
			testGrid(coord, axis) = double(rand()) / double(RAND_MAX);
		});
	}

	// Copy test
	VectorGrid<double> copyGrid = testGrid;

	EXPECT_TRUE(copyGrid.isGridMatched(testGrid));

	// Same values test
	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
		{
			EXPECT_EQ(copyGrid(coord, axis), testGrid(coord, axis));
		});
	}

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), copyGrid.size(axis), [&](const Vec3i& coord)
		{
			copyGrid(coord, axis) += 5.;
		});
	}

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
		{
			EXPECT_NE(copyGrid(coord, axis), testGrid(coord, axis));
		});
	}

	// Transform test
	EXPECT_EQ(testGrid.dx(), xform.dx());
	EXPECT_TRUE(testGrid.offset() == xform.offset());
	EXPECT_TRUE(testGrid.xform() == xform);
}

TEST(VECTOR_GRID_TESTS, CENTER_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::NODE);
}

TEST(VECTOR_GRID_TESTS, EDGE_SAMPLE_TEST)
{
	testSampleType(VectorGridSettings::SampleType::EDGE);
}

TEST(VECTOR_GRID_TESTS, INITIALIZE_TEST)
{
	double dx = .01;
	Vec3d origin = Vec3d::Random();
	Vec3i cellSize = Vec3i(10, 20, 30);

	double value = 1.;

	Transform xform(dx, origin);
	VectorGrid<double> testGrid(xform, cellSize, Vec3d::Constant(value));

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& cell)
		{
			EXPECT_EQ(testGrid(cell, axis), value);
		});
	}
}

// Min/max tests
TEST(VECTOR_GRID_TESTS, MIN_MAX_TEST)
{
	double dx = .01;
	Vec3d origin = Vec3d::Random();
	Vec3i cellSize = Vec3i(10, 20, 30);

	Transform xform(dx, origin);
	VectorGrid<double> testGrid(xform, cellSize, Vec3d::Ones());

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-100., 100.);

	for (int axis : {0, 1, 2})
	{
		double minValue = std::numeric_limits<double>::max();
		double maxValue = std::numeric_limits<double>::lowest();

		forEachVoxelRange(Vec3i::Zero(), testGrid.size(axis), [&](const Vec3i& coord)
		{
			double value = distribution(generator);
			minValue = std::min(value, minValue);
			maxValue = std::max(value, maxValue);
			testGrid(coord, axis) = value;
		});

		EXPECT_EQ(minValue, testGrid.grid(axis).minValue());
		EXPECT_EQ(maxValue, testGrid.grid(axis).maxValue());

		auto minMaxPair = testGrid.grid(axis).minAndMaxValue();

		EXPECT_EQ(minValue, minMaxPair.first);
		EXPECT_EQ(maxValue, minMaxPair.second);
	}
}

static void readWriteTest(const VectorGridSettings::SampleType sampleType)
{
	double dx = .01;
	Vec3d origin = Vec3d::Random();
	Vec3i cellSize = Vec3i(10, 20, 30);

	Transform xform(dx, origin);

	auto testFunc = [](const Vec3d& point) -> Vec3d
	{
		return Vec3d(std::sin(PI * point[0]), std::sin(2. * PI * point[1]), std::sin(4. * PI * point[2]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec3d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Zero(), testGrid.grid(axis).size(), [&](const Vec3i& coord)
		{
			Vec3d point = testGrid.indexToWorld(coord.cast<double>(), axis);
			double val = testFunc(point)[axis];
			double storedVal = testGrid(coord, axis);

			EXPECT_EQ(val, storedVal);
		});
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::NODE);
}

TEST(VECTOR_GRID_TESTS, EDGE_READ_WRITE_TEST)
{
	readWriteTest(VectorGridSettings::SampleType::EDGE);
}

// Component interpolation test

static double componentInterpolationErrorTest(const Transform& xform, const Vec3i& cellSize, const VectorGridSettings::SampleType sampleType)
{
	auto testFunc = [](const Vec3d& point) -> Vec3d
	{
		return Vec3d(std::sin(PI * point[0]), std::sin(PI * point[1]), std::sin(PI * point[2]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i coord = testGrid.grid(axis).unflatten(cellIndex);
				Vec3d point = testGrid.indexToWorld(coord.cast<double>(), axis);
				testGrid(coord, axis) = testFunc(point)[axis];
			}
		});
	}

	double error = 0;
	for (int axis : {0, 1, 2})
	{
		double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount(), tbbLightGrainSize), double(0),
			[&](const tbb::blocked_range<int>& range, double error) -> double
			{
				for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
				{
					Vec3i coord = testGrid.grid(axis).unflatten(cellIndex);

					if (coord[0] == testGrid.size(axis)[0] - 1 || coord[1] == testGrid.size(axis)[1] - 1 || coord[2] == testGrid.size(axis)[2] - 1)
						continue;

					Vec3d startPoint = coord.cast<double>();
					Vec3d endPoint = (coord + Vec3i::Ones()).cast<double>();

					Vec3d point;
					for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
						for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
                            for (point[2] = startPoint[2]; point[2] < endPoint[2]; point[2] += .2)
                            {
                                Vec3d worldPoint = testGrid.indexToWorld(point, axis);

                                double localError = std::fabs(testGrid.triLerp(worldPoint, axis) - testFunc(worldPoint)[axis]);
                                error = std::max(error, localError);
                            }
				}

				return error;
			},
			[](double a, double b) -> double
			{
				return std::max(a, b);
			}
			);

		error = std::max(error, localError);
	}

	return error;
}

static void componentInterpolationTest(const VectorGridSettings::SampleType sampleType)
{
	Vec3d origin = Vec3d::Zero();

	int base_size = 8;
	Vec3i cellSize = Vec3i::Constant(base_size);

	double dx = 1. / double(base_size);

	int testSize = 3;
	std::vector<double> errors;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		double localError = componentInterpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.85);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::NODE);
}

TEST(VECTOR_GRID_TESTS, EDGE_COMPONENT_INTERPLOTATION_TEST)
{
	componentInterpolationTest(VectorGridSettings::SampleType::EDGE);
}

// Vector interpolation test

static double vectorInterpolationErrorTest(const Transform& xform, const Vec3i& cellSize, const VectorGridSettings::SampleType sampleType)
{
	auto testFunc = [](const Vec3d& point) -> Vec3d
	{
		return Vec3d(std::sin(PI * point[0]), std::sin(PI * point[1]), std::sin(PI * point[2]));
	};

	VectorGrid<double> testGrid(xform, cellSize, sampleType);

	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, testGrid.grid(axis).voxelCount()), [&](const tbb::blocked_range<int>& range)
        {
            for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
            {
                Vec3i coord = testGrid.grid(axis).unflatten(cellIndex);
                Vec3d point = testGrid.indexToWorld(coord.cast<double>(), axis);
                testGrid(coord, axis) = testFunc(point)[axis];
            }
        });
	}

	double error = 0;

	for (int axis : {0, 1, 2})
	{
		forEachVoxelRange(Vec3i::Ones(), cellSize - Vec3i::Ones(), [&](const Vec3i& coord)
		{
			Vec3d startPoint = coord.cast<double>();
			Vec3d endPoint = (coord + Vec3i::Ones()).cast<double>();

			Vec3d point;
			for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
				for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
					for (point[2] = startPoint[2]; point[2] < endPoint[2]; point[2] += .2)
					{
						Vec3d worldPoint = testGrid.indexToWorld(point, axis);

						double localError = (testGrid.triLerp(worldPoint) - testFunc(worldPoint)).norm();
						error = std::max(error, localError);
					}
		});
	}

	return error;
}

static void vectorInterpolationTest(const VectorGridSettings::SampleType sampleType)
{
	Vec3d origin = Vec3d::Zero();

	int base_size = 8;
	Vec3i cellSize = Vec3i::Constant(base_size);

	double dx = 1. / double(base_size);

	int testSize = 3;
	std::vector<double> errors;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i localCellSize = int(std::pow(2, testIndex)) * cellSize;
		double localDx = dx / std::pow(2., testIndex);
		Transform xform(localDx, origin);

		double localError = vectorInterpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.8);
	}
}

TEST(VECTOR_GRID_TESTS, CENTER_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::CENTER);
}

TEST(VECTOR_GRID_TESTS, STAGGERED_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::STAGGERED);
}

TEST(VECTOR_GRID_TESTS, NODE_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::NODE);
}

TEST(VECTOR_GRID_TESTS, EDGE_VECTOR_INTERPLOTATION_TEST)
{
	vectorInterpolationTest(VectorGridSettings::SampleType::EDGE);
}