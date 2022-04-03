#include "gtest/gtest.h"

#include "SparseUniformGrid.h"
#include "UniformGrid.h"
#include "Utilities.h"

using namespace FluidSim3D;

//
// SparseTile tests
//

TEST(SPARSE_TILE_TESTS, DEFAULT_CONSTRUCTOR_TEST)
{
	SparseTile<int, 10> tile;
	EXPECT_TRUE(tile.constant());
}

TEST(SPARSE_TILE_TESTS, CONSTRUCTOR_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	EXPECT_TRUE(tile.constant());

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});
}

TEST(SPARSE_TILE_TESTS, EXPAND_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	tile.expand();
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});

	const double value2 = value * value;
	tile.makeConstant(value2);

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value2);
	});
}

TEST(SPARSE_TILE_TESTS, COLLAPSE_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);
	tile.expand();
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});

	tile.collapseIfConstant();

	EXPECT_TRUE(tile.constant());

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), value);
	});
}

TEST(SPARSE_TILE_TESTS, WRITE_TILE_TEST)
{
	const double value = PI;
	SparseTile<double, 10> tile(value);

	const double value2 = value * value;
	tile.setVoxel(Vec3i::Zero(), value2);
	EXPECT_FALSE(tile.constant());

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		if (cell == Vec3i::Zero())
		{
			EXPECT_EQ(tile.getVoxel(cell), value2);
		}
		else
		{
			EXPECT_EQ(tile.getVoxel(cell), value);
		}		
	});

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		tile.setVoxel(cell, cell[0] + cell[1] * 10 + cell[2] * 100);
	});

	forEachVoxelRangeReverse(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_EQ(tile.getVoxel(cell), cell[0] + cell[1] * 10 + cell[2] * 100);
	});
}

TEST(SPARSE_TILE_TESTS, FLATTEN_ROUND_TRIP_TEST)
{
	SparseTile<double, 10> tile;
	tile.expand();

	forEachVoxelRange(Vec3i::Zero(), Vec3i::Constant(10), [&](const Vec3i& cell)
	{
		EXPECT_TRUE(tile.unflatten(tile.flatten(cell)) == cell);
	});
}

//
// SparseUniformGrid tests
//

TEST(SPARSE_UNIFORM_GRID_TESTS, DEFAULT_CONSTRUCTOR_TEST)
{
	SparseUniformGrid<int, 10> sparseGrid;
	EXPECT_TRUE(sparseGrid.tileSize() == Vec3i::Zero());
	EXPECT_TRUE(sparseGrid.gridSize() == Vec3i::Zero());
}

TEST(SPARSE_UNIFORM_GRID_TESTS, CONSTRUCTOR_TEST)
{
	Vec3i size(100, 200, 300);
	SparseUniformGrid<double, 10> sparseGrid(size);

	EXPECT_TRUE(sparseGrid.gridSize() == size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.gridSize(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(sparseGrid.getVoxel(cell), double(0));
	});

	EXPECT_TRUE(sparseGrid.tileSize() == Vec3i(10, 20, 30));

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.tile(tileCoord).constant());
	});

	EXPECT_TRUE(sparseGrid.tileCount() == 10 * 20 * 30);
}


TEST(SPARSE_UNIFORM_GRID_TESTS, VALUE_CONSTRUCTOR_TEST)
{
	Vec3i size(100, 200, 300);
	const double value = PI;
	SparseUniformGrid<double, 10> sparseGrid(size, value);

	EXPECT_TRUE(sparseGrid.gridSize() == size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.gridSize(), [&](const Vec3i& cell)
	{
		EXPECT_EQ(sparseGrid.getVoxel(cell), value);
	});

	EXPECT_TRUE(sparseGrid.tileSize() == Vec3i(10, 20, 30));

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.tile(tileCoord).constant());
	});

	EXPECT_TRUE(sparseGrid.tileCount() == 10 * 20 * 30);
}

TEST(SPARSE_UNIFORM_GRID_TESTS, TILE_FLATTEN_ROUNDTRIP_TEST)
{
	Vec3i size(100, 200, 300);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.unflattenTileIndex(sparseGrid.flattenTileIndex(tileCoord)) == tileCoord);
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, TILE_VOXEL_RANGE_TEST)
{
	Vec3i size(111, 222, 333);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		auto tileRange = sparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec3i& cell)
		{
			EXPECT_TRUE(cell[0] < size[0] && cell[1] < size[1] && cell[2] < size[2]);
		});
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, WRITE_READ_TEST)
{
	Vec3i size(111, 222, 333);
	SparseUniformGrid<double, 10> sparseGrid(size);

	UniformGrid<double> uniformGrid(size);

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		uniformGrid(cell) = cell[0] + cell[1] * size[0] + cell[2] * size[0] * size[2];
		sparseGrid.setVoxel(cell, uniformGrid(cell));
	});

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		EXPECT_EQ(uniformGrid(cell), sparseGrid.getVoxel(cell));
	});

	SparseUniformGrid<double, 10> tileWrittenSparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), tileWrittenSparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		auto tileRange = tileWrittenSparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec3i& cell)
		{
			tileWrittenSparseGrid.setVoxel(cell, uniformGrid(cell));
		});
	});

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		EXPECT_EQ(uniformGrid(cell), tileWrittenSparseGrid.getVoxel(cell));
	});

	forEachVoxelRange(Vec3i::Zero(), tileWrittenSparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		auto tileRange = tileWrittenSparseGrid.tileVoxelRange(tileCoord);
		forEachVoxelRange(tileRange.first, tileRange.second, [&](const Vec3i& cell)
		{
			EXPECT_EQ(uniformGrid(cell), tileWrittenSparseGrid.getVoxel(cell));
		});
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, WRITE_EXPAND_TEST)
{
	Vec3i size(111, 222, 333);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0] * cell[2] * size[0] * size[1]);
	});

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, COLLAPSE_TILE_TEST)
{
	Vec3i size(111, 222, 333);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		sparseGrid.expandTile(tileCoord);
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});

	sparseGrid.collapseTiles();

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0] + cell[2] * size[0] * size[1]);
	});

	sparseGrid.collapseTiles();

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_FALSE(sparseGrid.isTileConstant(tileCoord));
	});
}

TEST(SPARSE_UNIFORM_GRID_TESTS, RESIZE_TEST)
{
	Vec3i size(111, 222, 333);
	SparseUniformGrid<double, 10> sparseGrid(size);

	forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& cell)
	{
		sparseGrid.setVoxel(cell, cell[0] + cell[1] * size[0] + cell[2] * size[0] * size[1]);
	});

	Vec3i largerSize = 2 * size;
	sparseGrid.resize(largerSize);

	EXPECT_TRUE(sparseGrid.gridSize() == largerSize);

	forEachVoxelRange(Vec3i::Zero(), sparseGrid.tileSize(), [&](const Vec3i& tileCoord)
	{
		EXPECT_TRUE(sparseGrid.isTileConstant(tileCoord));
	});
}