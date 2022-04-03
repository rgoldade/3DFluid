#ifndef FLUIDSIM3D_SPARSE_UNIFORM_GRID_H
#define FLUIDSIM3D_SPARSE_UNIFORM_GRID_H

#include "tbb/blocked_range.h"
#include "tbb/parallel_reduce.h"

#include "GridUtilities.h"
#include "Utilities.h"

namespace FluidSim3D
{

template <typename T, int TILE_SIZE>
class SparseTile
{
public:
	SparseTile() : SparseTile(T(0))
	{}

	SparseTile(const T& value)
		: myGrid(VectorXt<T>::Constant(1, value))
	{}

	bool constant() const
	{
		return myGrid.size() == 1;
	}

	void makeConstant(const T& value)
	{
		myGrid = VectorXt<T>::Constant(1, value);
	}

	void expand()
	{
		if (constant())
		{
			assert(myGrid.size() == 1);
			const T value = myGrid(0);
			myGrid = VectorXt<T>::Constant(std::pow(TILE_SIZE, 3), value);
		}
	}

	void collapseIfConstant()
	{
		if (!constant())
		{
			if (myGrid.maxCoeff() == myGrid.minCoeff())
			{
				myGrid = VectorXt<T>::Constant(1, myGrid(0));
			}
		}
	}

	FORCE_INLINE T getVoxel(const Vec3i& coord) const
	{
		for (int axis : {0, 1, 2})
		{
			assert(coord[axis] >= 0 && coord[axis] < TILE_SIZE);
		}

		if (constant())
		{
			return myGrid(0);
		}
		else
		{
			return myGrid[flatten(coord)];
		}
	}

	FORCE_INLINE void setVoxel(const Vec3i& coord, const T& value)
	{
		for (int axis : {0, 1, 2})
		{
			assert(coord[axis] >= 0 && coord[axis] < TILE_SIZE);
		}
		
		if (constant())
		{
			// TODO: add lock to prevent race conditions
			expand();
		}
			
		myGrid[flatten(coord)] = value;
	}

	FORCE_INLINE int flatten(const Vec3i& coord) const { return coord[2] + TILE_SIZE * coord[1] + TILE_SIZE * TILE_SIZE * coord[0]; }

	FORCE_INLINE Vec3i unflatten(int index) const
	{
		assert(index >= 0 && index < std::pow(TILE_SIZE, 3));

		Vec3i coord;
		coord[2] = index % TILE_SIZE;

		index -= coord[2];
		index /= TILE_SIZE;

		coord[1] = index % TILE_SIZE;

		index -= coord[1];
		index /= TILE_SIZE;

		coord[0] = index;

		return coord;
	}

	T maxValue() const
	{
		return myGrid.maxCoeff();
	}

	T minValue() const
	{
		return myGrid.minCoeff();
	}

private:
	VectorXt<T> myGrid;
};


template <typename T, int TILE_SIZE=16>
class SparseUniformGrid
{
public:

	SparseUniformGrid()
	: myGridSize(Vec3i::Zero())
	, myTileSize(Vec3i::Zero())
	{}

	SparseUniformGrid(const Vec3i& size) : SparseUniformGrid(size, T(0)) {}

	SparseUniformGrid(const Vec3i& size, const T& value)
	: myGridSize(size)
	, myTileSize((size.cast<double>() / double(TILE_SIZE)).array().ceil().cast<int>())
	, myTiles(tileCount(), SparseTile<T, TILE_SIZE>(value))
	{
		for (int axis : {0, 1, 2})
		{
			assert(size[axis] >= 0);
		}
	}

	int tileCount() const { return myTileSize[0] * myTileSize[1] * myTileSize[2]; }

	FORCE_INLINE int flattenTileIndex(const Vec3i& tileIndex) const
	{
		for (int axis : {0, 1, 2})
		{
			assert(tileIndex[axis] >= 0 && tileIndex[axis] < myTileSize[axis]);
		}

		return tileIndex[2] + myTileSize[2] * tileIndex[1] + myTileSize[2] * myTileSize[1] * tileIndex[0];
	}

	FORCE_INLINE Vec3i unflattenTileIndex(int flatTileIndex) const
	{
		assert(flatTileIndex >= 0 && flatTileIndex < myTileSize[0] * myTileSize[1] * myTileSize[2]);

		Vec3i tileCoord;
		tileCoord[2] = flatTileIndex % myTileSize[2];

		flatTileIndex /= myTileSize[2];

		tileCoord[1] = flatTileIndex % myTileSize[1];

		flatTileIndex /= myTileSize[1];

		tileCoord[0] = flatTileIndex;

		return tileCoord;
	}

	const SparseTile<T, TILE_SIZE>& tile(const Vec3i& tileIndex) const
	{
		for (int axis : {0, 1, 2})
		{
			assert(tileIndex[axis] >= 0 && tileIndex[axis] < myTileSize[axis]);
		}

		return myTiles[flattenTileIndex(tileIndex)];
	}

	SparseTile<T, TILE_SIZE>& tile(const Vec3i& tileIndex)
	{
		for (int axis : {0, 1, 2})
		{
			assert(tileIndex[axis] >= 0 && tileIndex[axis] < myTileSize[axis]);
		}

		return myTiles[flattenTileIndex(tileIndex)];
	}

	std::pair<Vec3i, Vec3i> tileVoxelRange(const Vec3i& tileIndex)
	{
		std::pair<Vec3i, Vec3i> voxelRange;
		voxelRange.first = tileIndex * TILE_SIZE;
		voxelRange.second = ((tileIndex + Vec3i::Ones()) * TILE_SIZE).cwiseMin(myGridSize);

		return voxelRange;
	}

	T getVoxel(const Vec3i& voxelIndex) const
	{
		// Get tile index
		Vec3i tileIndex = voxelIndex / TILE_SIZE;

		Vec3i localVoxelIndex = voxelIndex - tileIndex * TILE_SIZE;

		return tile(tileIndex).getVoxel(localVoxelIndex);
	}

	void setVoxel(const Vec3i& voxelIndex, const T& value)
	{
		// Get tile index
		Vec3i tileIndex = voxelIndex / TILE_SIZE;

		auto& localTile = tile(tileIndex);
		if (localTile.constant())
		{
			// Expand constant tile
			localTile.expand();
		}

		Vec3i localVoxelIndex = voxelIndex - tileIndex * TILE_SIZE;

		localTile.setVoxel(localVoxelIndex, value);
	}

	bool isTileConstant(const Vec3i& tileIndex) const
	{
		return tile(tileIndex).constant();
	}

	void expandTile(const Vec3i& tileIndex)
	{
		tile(tileIndex).expand();
	}

	void collapseTiles()
	{
		for (auto& localTile : myTiles)
		{
			localTile.collapseIfConstant();
		}
	}

	const Vec3i& gridSize() const
	{
		return myGridSize;
	}

	const Vec3i& tileSize() const
	{
		return myTileSize;
	}	

	void resize(const Vec3i& size)
	{
		resize(size, T(0));
	}

	void resize(const Vec3i& size, const T& value)
	{
		for (int axis : {0, 1, 2})
		{
			assert(size[axis] >= 0);
		}

		myGridSize = size;
		myTileSize = (size.cast<double>() / double(TILE_SIZE)).array().ceil().cast<int>();
		myTiles.clear();
		myTiles.resize(tileCount(), SparseTile<T, TILE_SIZE>(value));
	}

	T maxValue() const
	{
		return tbb::parallel_reduce(tbb::blocked_range<int>(0, tileCount()), std::numeric_limits<T>::lowest(), [&](const tbb::blocked_range<int>& range, T localMaxValue)
				{
					for (int flatTileIndex = range.begin(); flatTileIndex != range.end(); ++flatTileIndex)
					{
						Vec3i tileCoord = unflattenTileIndex(flatTileIndex);
						localMaxValue = std::max(localMaxValue, tile(tileCoord).maxValue());
					}

					return localMaxValue;
				},
				[&](const T a, const T b)
				{
					return std::max(a, b);
				});
	}

	T minValue() const
	{
		return tbb::parallel_reduce(tbb::blocked_range<int>(0, tileCount()), std::numeric_limits<T>::max(), [&](const tbb::blocked_range<int>& range, T localMinValue)
			{
				for (int flatTileIndex = range.begin(); flatTileIndex != range.end(); ++flatTileIndex)
				{
					Vec3i tileCoord = unflattenTileIndex(flatTileIndex);
					localMinValue = std::min(localMinValue, tile(tileCoord).minValue());
				}

				return localMinValue;
			},
			[&](const T a, const T b)
			{
				return std::min(a, b);
			});

	}

	std::pair<T, T> minAndMaxValue() const
	{
		return std::make_pair<T, T>(minValue(), maxValue());
	}

private:

	Vec3i myGridSize;
	Vec3i myTileSize;

	std::vector<SparseTile<T, TILE_SIZE>> myTiles;
};


}

#endif
