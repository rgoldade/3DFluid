#include "gtest/gtest.h"

#include "GridUtilities.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim3D;

template<typename T, typename ValFunc>
static void fillUniformGrid(const ValFunc& valFunc, UniformGrid<T>& grid)
{
	forEachVoxelRange(Vec3i::Zero(), grid.size(), [&](const Vec3i& cell)
	{
		grid(cell) = valFunc(cell);
	});
}

TEST(UNIFORM_GRID_TESTS, DEFAULT_CONSTRUCTOR_SIZE_TEST)
{
	UniformGrid<int> testGrid;
	EXPECT_TRUE(testGrid.size() == Vec3i::Zero());
}

TEST(UNIFORM_GRID_TESTS, CONSTRUCTOR_SIZE_TEST)
{
	Vec3i size(10, 20, 30);
	UniformGrid<int> testGrid(size);

	EXPECT_TRUE(testGrid.size() == size);
}

TEST(UNIFORM_GRID_TESTS, CONSTRUCTOR_SIZE_AND_VALUE_TEST)
{
	Vec3i size(10, 20, 30);
    int val = 10;
	UniformGrid<int> testGrid(size, val);

	EXPECT_TRUE(testGrid.size() == size);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(val, testGrid(cell));
		EXPECT_EQ(val, testGrid(cell[0], cell[1], cell[2]));
	});
}

TEST(UNIFORM_GRID_TESTS, STORAGE_ROUND_TRIP)
{
	Vec3i size(10, 20, 30);

	auto valFunc = [](const Vec3i& cell) -> double
	{
		Vec3d vec = cell.cast<double>();
		return vec[0] * std::pow(vec[1], 2) * std::pow(vec[2], 3);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
		EXPECT_EQ(valFunc(cell), testGrid(cell[0], cell[1], cell[2]));
	});
}

TEST(UNIFORM_GRID_TESTS, CLEAR_TEST)
{
	Vec3i size(10, 20, 30);
	UniformGrid<double> testGrid(size);

	EXPECT_FALSE(testGrid.empty());

	testGrid.clear();
	EXPECT_TRUE(testGrid.empty());
	EXPECT_TRUE(testGrid.size() == Vec3i::Zero());
}

TEST(UNIFORM_GRID_TESTS, RESIZE_LARGER_TEST)
{
	Vec3i size(10, 20, 30);

	auto valFunc = [](const Vec3i& cell) -> double
	{
		Vec3d vec = cell.cast<double>();
		return vec[0] * std::pow(vec[1], 2) * std::pow(vec[2], 3);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	// Resize grid
	Vec3i expandSize = 2 * size;
	testGrid.resize(expandSize);
	fillUniformGrid<double>(valFunc, testGrid);

	EXPECT_TRUE(testGrid.size() == expandSize);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESIZE_SMALLER_TEST)
{
	Vec3i size(10, 20, 30);

	auto valFunc = [](const Vec3i& cell) -> double
	{
		Vec3d vec = cell.cast<double>();
		return vec[0] * std::pow(vec[1], 2) * std::pow(vec[2], 3);
	};

	UniformGrid<double> testGrid(size);
	fillUniformGrid<double>(valFunc, testGrid);

	// Resize grid
	Vec3i shrinkSize(size[0] / 2, size[1] / 2, size[2] / 2);
	testGrid.resize(shrinkSize);
	fillUniformGrid<double>(valFunc, testGrid);

	EXPECT_TRUE(testGrid.size() == shrinkSize);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(valFunc(cell), testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESIZE_VALUE_TEST)
{
	Vec3i size(10, 20, 30);
	double startValue = 10.;
	UniformGrid<double> testGrid(size, startValue);

	// Resize grid
	Vec3i expandSize = 2 * size;
	double expandValue = 100.;
	testGrid.resize(expandSize, expandValue);

	EXPECT_TRUE(testGrid.size() == expandSize);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(expandValue, testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, RESET_TEST)
{
	Vec3i size(10, 20, 30);
	double startValue = 10.;
	UniformGrid<double> testGrid(size, startValue);

	double resetValue = 100.;
	testGrid.reset(resetValue);

	EXPECT_TRUE(testGrid.size() == size);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(resetValue, testGrid(cell));
	});
}

TEST(UNIFORM_GRID_TESTS, VOXEL_COUNT_TEST)
{
	Vec3i size(10, 20, 30);
	UniformGrid<double> testGrid(size);

	EXPECT_EQ(size[0] * size[1] * size[2], testGrid.voxelCount());
}

TEST(UNIFORM_GRID_TESTS, FLATTEN_UNFLATTEN_TEST)
{
	int testSize = 100;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i size = (20. * (Vec3d::Random() + Vec3d::Ones())).cast<int>();

		if (size[0] <= 0 || size[1] <= 0 && size[2] <= 0)
			continue;

		UniformGrid<double> testGrid(size);

		forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
		{
			EXPECT_TRUE(testGrid.unflatten(testGrid.flatten(cell)) == cell);
		});
	}
}

TEST(UNIFORM_GRID_TESTS, FLATTEN_COVERAGE_TEST)
{
	int testSize = 100;
	double testVal = PI;
	for (int testIndex = 0; testIndex < testSize; ++testIndex)
	{
		Vec3i size = (20. * (Vec3d::Random() + Vec3d::Ones())).cast<int>();

		if (size[0] <= 0 || size[1] <= 0 && size[2] <= 0)
			continue;

		UniformGrid<double> testGrid(size);

		for (int flatIndex = 0; flatIndex < size.prod(); ++flatIndex)
		{
			Vec3i cell = testGrid.unflatten(flatIndex);
			testGrid(cell) = testVal;
		}

		forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
		{
			EXPECT_EQ(testGrid(cell), testVal);
		});
	}
}

TEST(UNIFORM_GRID_TESTS, COPY_GRID_TEST)
{
	Vec3i size(10, 20, 30);
	UniformGrid<int> testGrid(size, 10);

	UniformGrid<int> copyGrid = testGrid;

	EXPECT_TRUE(testGrid.size() == copyGrid.size());

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(testGrid(cell), copyGrid(cell));
	});

	copyGrid.reset(15);

	forEachVoxelRange(Vec3i::Zero(), testGrid.size(), [&](const Vec3i& cell)
	{
		EXPECT_NE(testGrid(cell), copyGrid(cell));
	});
}