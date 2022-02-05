#include <random>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "gtest/gtest.h"

#include "GridUtilities.h"
#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"

using namespace FluidSim3D;

TEST(SCALAR_GRID_TESTS, DEFAULT_CONSTRUCTOR_SIZE_TEST)
{
	ScalarGrid<double> testGrid;
	EXPECT_TRUE(testGrid.size() == Vec3i::Zero());
}

static void testSampleType(const ScalarGridSettings::SampleType sampleType)
{
	double dx = 2. / 25.;
	Vec3d origin = Vec3d::Constant(-1);
	Vec3i cellSize = Vec3i::Constant(25);

	Transform xform(dx, origin);

	ScalarGrid<double> testGrid(xform, cellSize, sampleType);

	EXPECT_EQ(testGrid.sampleType(), sampleType);

	// Test sample count related to the grid size 
	switch (sampleType)
	{
		case ScalarGridSettings::SampleType::CENTER:
		{
			EXPECT_TRUE(testGrid.size() == cellSize);
			break;
		}
		case ScalarGridSettings::SampleType::XFACE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(1, 0, 0)).eval());
			break;
		}
		case ScalarGridSettings::SampleType::YFACE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(0, 1, 0)).eval());
			break;
		}
        case ScalarGridSettings::SampleType::ZFACE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(0, 0, 1)).eval());
			break;
		}
		case ScalarGridSettings::SampleType::XEDGE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(0, 1, 1)).eval());
			break;
		}
		case ScalarGridSettings::SampleType::YEDGE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(1, 0, 1)).eval());
			break;
		}
        case ScalarGridSettings::SampleType::ZEDGE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i(1, 1, 0)).eval());
			break;
		}

		case ScalarGridSettings::SampleType::NODE:
		{
			EXPECT_TRUE(testGrid.size() == (cellSize + Vec3i::Ones()).eval());
		}
	}

	// Test index-to-world and back
	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		Vec3d indexPoint = testGrid.worldToIndex(testGrid.indexToWorld(coord.cast<double>()));
		EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
		EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
        EXPECT_TRUE(isNearlyEqual(indexPoint[2], double(coord[2]), 1e-5, false));
	});
		
	// Test sampling
	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		Vec3d worldPoint;
		switch (sampleType)
		{
			case ScalarGridSettings::SampleType::CENTER:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d::Constant(.5));
				break;
			}
			case ScalarGridSettings::SampleType::XFACE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., .5, .5));
				break;
			}
			case ScalarGridSettings::SampleType::YFACE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, 0., .5));
				break;
			}
			case ScalarGridSettings::SampleType::ZFACE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, .5, 0.));
				break;
			}
			case ScalarGridSettings::SampleType::XEDGE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(.5, 0., 0.));
				break;
			}
			case ScalarGridSettings::SampleType::YEDGE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., .5, 0.));
				break;
			}
			case ScalarGridSettings::SampleType::ZEDGE:
			{
				worldPoint = origin + dx * (coord.cast<double>() + Vec3d(0., 0., .5));
				break;
			}			
			case ScalarGridSettings::SampleType::NODE:
			{
				worldPoint = origin + dx * coord.cast<double>();
			}
		}

		Vec3d indexPoint = testGrid.worldToIndex(worldPoint);

		EXPECT_TRUE(isNearlyEqual(indexPoint[0], double(coord[0]), 1e-5, false));
		EXPECT_TRUE(isNearlyEqual(indexPoint[1], double(coord[1]), 1e-5, false));
        EXPECT_TRUE(isNearlyEqual(indexPoint[2], double(coord[2]), 1e-5, false));
	});

    forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
    {
        testGrid(coord) = double(rand()) / double(RAND_MAX);
    });

	// Copy test
	ScalarGrid<double> copyGrid = testGrid;

	EXPECT_TRUE(copyGrid.isGridMatched(testGrid));

	// Same values test
	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		EXPECT_EQ(copyGrid(coord), testGrid(coord));
	});

	copyGrid += 5.;

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		EXPECT_NE(copyGrid(coord), testGrid(coord));
	});

	testGrid = copyGrid;

	// Same values test again
	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		EXPECT_EQ(copyGrid(coord), testGrid(coord));
	});

	copyGrid *= 10.;

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& coord)
	{
		EXPECT_NE(copyGrid(coord), testGrid(coord));
	});

	// Transform test
	EXPECT_EQ(testGrid.dx(), xform.dx());
	EXPECT_TRUE(testGrid.offset() == xform.offset());
	EXPECT_TRUE(testGrid.xform() == xform);
}

TEST(SCALAR_GRID_TESTS, CENTER_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, ZFACE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::ZFACE);
}

TEST(SCALAR_GRID_TESTS, XEDGE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::XEDGE);
}

TEST(SCALAR_GRID_TESTS, YEDGE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::YEDGE);
}

TEST(SCALAR_GRID_TESTS, ZEDGE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::ZEDGE);
}

TEST(SCALAR_GRID_TESTS, NODE_SAMPLE_TEST)
{
	testSampleType(ScalarGridSettings::SampleType::NODE);
}

TEST(SCALAR_GRID_TESTS, INITIALIZE_TEST)
{
	double dx = 2. / 25.;
	Vec3d origin = Vec3d::Constant(-1);
	Vec3i cellSize = Vec3i::Constant(25);

	double value = 1.;

	Transform xform(dx, origin);
	ScalarGrid<double> testGrid(xform, cellSize, value);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(testGrid(cell), value);
	});
}

// Min/max tests
TEST(SCALAR_GRID_TESTS, MIN_MAX_TEST)
{
	double dx = 2. / 25.;
	Vec3d origin = Vec3d::Constant(-1);
	Vec3i cellSize = Vec3i::Constant(25);

	Transform xform(dx, origin);
	ScalarGrid<double> testGrid(xform, cellSize, 1.);

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(-100., 100.);

	double minValue = std::numeric_limits<double>::max();
	double maxValue = std::numeric_limits<double>::lowest();

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		double value = distribution(generator);
		minValue = std::min(value, minValue);
		maxValue = std::max(value, maxValue);
		testGrid(cell) = value;
	});

	EXPECT_EQ(minValue, testGrid.minValue());
	EXPECT_EQ(maxValue, testGrid.maxValue());

	auto minMaxPair = testGrid.minAndMaxValue();

	EXPECT_EQ(minValue, minMaxPair.first);
	EXPECT_EQ(maxValue, minMaxPair.second);
}

static void readWriteTest(const ScalarGridSettings::SampleType sampleType)
{
	Vec3d origin = Vec3d::Constant(-1);
	Vec3i cellSize = Vec3i(16, 32, 48);

	double dx = 1. / 16.;

	Transform xform(dx, origin);

	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec3d& point) -> double
	{
		return std::sin(PI * point[0]) * std::sin(2. * PI * point[1]) * std::sin(4. * PI * point[2]);
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i coord = grid.unflatten(cellIndex);
			Vec3d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	forEachVoxelRange(Vec3i::Zero(), grid.size(), [&](const Vec3i& coord)
	{
		Vec3d point = grid.indexToWorld(coord.cast<double>());
		double val = testFunc(point);
		double storedVal = grid(coord);

		EXPECT_EQ(val, storedVal);
	});
}

TEST(SCALAR_GRID_TESTS, CENTER_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, ZFACE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::ZFACE);
}

TEST(SCALAR_GRID_TESTS, XEDGE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::XEDGE);
}

TEST(SCALAR_GRID_TESTS, YEDGE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::YEDGE);
}

TEST(SCALAR_GRID_TESTS, ZEDGE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::ZEDGE);
}

TEST(SCALAR_GRID_TESTS, NODE_READ_WRITE_TEST)
{
	readWriteTest(ScalarGridSettings::SampleType::NODE);
}

// Interpolation test

static double interpolationErrorTest(const Transform& xform, const Vec3i& cellSize, const ScalarGridSettings::SampleType sampleType)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec3d& point) -> double
	{
		return std::sin(PI * point[0]) * std::sin(PI * point[1]) * std::sin(PI * point[2]);
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i coord = grid.unflatten(cellIndex);
			Vec3d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount()), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i coord = grid.unflatten(cellIndex);
				
				if (coord[0] == grid.size()[0] - 1 || coord[1] == grid.size()[1] - 1 || coord[2] == grid.size()[2] - 1)
					continue;

				Vec3d startPoint = coord.cast<double>();
				Vec3d endPoint = (coord + Vec3i::Ones()).cast<double>();

				Vec3d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
                        for (point[2] = startPoint[2]; point[2] < endPoint[2]; point[2] += .2)
                        {
                            Vec3d worldPoint = grid.indexToWorld(point);
                            double localError = std::fabs(grid.triLerp(worldPoint) - testFunc(worldPoint));
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

	return error;
}

static void interpolationTest(const ScalarGridSettings::SampleType sampleType)
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

		double localError = interpolationErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 3.8);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, ZFACE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::ZFACE);
}

TEST(SCALAR_GRID_TESTS, XEDGE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::XEDGE);
}

TEST(SCALAR_GRID_TESTS, YEDGE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::YEDGE);
}

TEST(SCALAR_GRID_TESTS, ZEDGE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::ZEDGE);
}

TEST(SCALAR_GRID_TESTS, NODE_INTERPLOTATION_TEST)
{
	interpolationTest(ScalarGridSettings::SampleType::NODE);
}

static double gradientErrorTest(const Transform& xform, const Vec3i& cellSize, const ScalarGridSettings::SampleType sampleType)
{
	ScalarGrid<double> grid(xform, cellSize, sampleType);
	auto testFunc = [](const Vec3d& point) -> double
	{
		return std::sin(PI * point[0]) * std::sin(PI * point[1]) * std::sin(PI * point[2]);
	};

	auto testFuncGradient = [](const Vec3d& point) -> Vec3d
	{
		return Vec3d(PI * std::cos(PI * point[0]) * std::sin(PI * point[1]) * std::sin(PI * point[2]),
						PI * std::sin(PI * point[0]) * std::cos(PI * point[1]) * std::sin(PI * point[2]),
						PI * std::sin(PI * point[0]) * std::sin(PI * point[1]) * std::cos(PI * point[2]));
	};

	tbb::parallel_for(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
		{
			Vec3i coord = grid.unflatten(cellIndex);
			Vec3d point = grid.indexToWorld(coord.cast<double>());
			grid(coord) = testFunc(point);
		}
	});

	double error = tbb::parallel_reduce(tbb::blocked_range<int>(0, grid.voxelCount(), tbbLightGrainSize), double(0),
		[&](const tbb::blocked_range<int>& range, double error) -> double
		{
			for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
			{
				Vec3i coord = grid.unflatten(cellIndex);

				if (coord[0] == grid.size()[0] - 1 || coord[1] == grid.size()[1] - 1 || coord[2] == grid.size()[2] - 1)
					continue;

				Vec3d startPoint = coord.cast<double>();
				Vec3d endPoint = (coord + Vec3i::Ones()).cast<double>();

				Vec3d point;
				for (point[0] = startPoint[0]; point[0] < endPoint[0]; point[0] += .2)
					for (point[1] = startPoint[1]; point[1] < endPoint[1]; point[1] += .2)
                        for (point[2] = startPoint[2]; point[2] < endPoint[2]; point[2] += .2)
                        {
                            Vec3d worldPoint = grid.indexToWorld(point);
                            Vec3d lerpGrad = grid.triLerpGradient(worldPoint);
                            Vec3d funcGrad = testFuncGradient(worldPoint);
                            double localError = (lerpGrad - funcGrad).norm();
                            error = std::max(localError, error);
                        }
			}

			return error;
		},
		[](double a, double b) -> double
		{
			return std::max(a, b);
		});

	return error;
}

static void gradientTest(const ScalarGridSettings::SampleType sampleType)
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

		double localError = gradientErrorTest(xform, localCellSize, sampleType);
		errors.push_back(localError);
	}

	for (int testIndex = 1; testIndex < testSize; ++testIndex)
	{
		double errorRatio = errors[testIndex - 1] / errors[testIndex];
		EXPECT_GT(errorRatio, 1.9);
	}
}

TEST(SCALAR_GRID_TESTS, CENTER_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::CENTER);
}

TEST(SCALAR_GRID_TESTS, XFACE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::XFACE);
}

TEST(SCALAR_GRID_TESTS, YFACE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::YFACE);
}

TEST(SCALAR_GRID_TESTS, ZFACE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::ZFACE);
}

TEST(SCALAR_GRID_TESTS, XEDGE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::XEDGE);
}

TEST(SCALAR_GRID_TESTS, YEDGE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::YEDGE);
}

TEST(SCALAR_GRID_TESTS, ZEDGE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::ZEDGE);
}

TEST(SCALAR_GRID_TESTS, NODE_GRADIENT_TEST)
{
	gradientTest(ScalarGridSettings::SampleType::NODE);
}
