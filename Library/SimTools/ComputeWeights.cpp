#include <array>

#include "tbb\tbb.h"
#include "ComputeWeights.h"

namespace FluidSim3D::SimTools
{

// Helper functions to compute face area fractions for the cut-cell weights.
// Taken from Christopher Batty's source code.
void rotateFaceValues(std::array<float, 4>& phiNodes)
{
	float firstNodePhi = phiNodes[0];
	for (int i = 0; i < 3; ++i)
		phiNodes[i] = phiNodes[i + 1];
	phiNodes[3] = firstNodePhi;
}

// Given four signed distance values (square corners), determine what fraction of the square is "inside" a surface.
// This boils down to simple geometric operations that compute the area of implicit triangles over the face.
// This is largely borrowed from Dr. Christopher Batty's sample code. (https://cs.uwaterloo.ca/~c2batty/).

// Node layout is CCW.
// 3 -- 2
// |    |
// 0 -- 1

float fractionInside(std::array<float, 4>& phiNodes)
{
	//float phiBottomLeft, float phiBottomRight, float phiTopLeft, float phiTopRight)
	int insideCount = (phiNodes[0] < 0. ? 1 : 0) + (phiNodes[1] < 0. ? 1 : 0) + (phiNodes[2] < 0. ? 1 : 0) + (phiNodes[3] < 0. ? 1 : 0);

	if (insideCount == 4)
		return 1.;
	else if (insideCount == 3)
	{
		// Rotate until the positive value is in the first position
		while (phiNodes[0] < 0.) rotateFaceValues(phiNodes);

		// Work out the area of the exterior triangle
		float side0 = 1. - lengthFraction(phiNodes[0], phiNodes[3]);
		float side1 = 1. - lengthFraction(phiNodes[0], phiNodes[1]);
		return 1. - 0.5 * side0 * side1;
	}
	else if (insideCount == 2)
	{
		// Rotate until a negative value is in the first position, and the next negative is in either slot 1 or 2.
		while (phiNodes[0] >= 0. || !(phiNodes[1] < 0. || phiNodes[2] < 0.)) rotateFaceValues(phiNodes);

		if (phiNodes[1] < 0) // The matching signs are adjacent
		{
			float sideLeft = lengthFraction(phiNodes[0], phiNodes[3]);
			float sideRight = lengthFraction(phiNodes[1], phiNodes[2]);
			return  0.5 * (sideLeft + sideRight);
		}
		else  // The matching signs are diagonally opposite
		{
			// Determine the centre point's sign to disambiguate this case
			float phiMiddle = 0.25 * (phiNodes[0] + phiNodes[1] + phiNodes[2] + phiNodes[3]);
			if (phiMiddle < 0.)
			{
				float area = 0.;

				// First triangle (top left)
				float side1 = 1. - lengthFraction(phiNodes[0], phiNodes[3]);
				float side3 = 1. - lengthFraction(phiNodes[2], phiNodes[3]);

				area += 0.5 * side1 * side3;

				// Second triangle (top right)
				float side2 = 1. - lengthFraction(phiNodes[2], phiNodes[1]);
				float side0 = 1. - lengthFraction(phiNodes[0], phiNodes[1]);

				area += 0.5 * side0 * side2;

				return 1. - area;
			}
			else
			{
				float area = 0.;

				// First triangle (bottom left)
				float side0 = lengthFraction(phiNodes[0], phiNodes[1]);
				float side1 = lengthFraction(phiNodes[0], phiNodes[3]);
				area += 0.5 * side0 * side1;

				// Second triangle (top right)
				float side2 = lengthFraction(phiNodes[2], phiNodes[1]);
				float side3 = lengthFraction(phiNodes[2], phiNodes[3]);
				area += 0.5 * side2 * side3;
				return area;
			}
		}
	}
	else if (insideCount == 1)
	{
		// Rotate until the negative value is in the first position
		while (phiNodes[0] >= 0) rotateFaceValues(phiNodes);

		// Work out the area of the interior triangle, and subtract from 1.
		float side0 = lengthFraction(phiNodes[0], phiNodes[3]);
		float side1 = lengthFraction(phiNodes[0], phiNodes[1]);
		return 0.5 * side0 * side1;
	}

	return 0.;
}

VectorGrid<float> computeGhostFluidWeights(const LevelSet& surface)
{
	VectorGrid<float> ghostFluidWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	for (int axis : {0, 1, 2})
	{
		Vec3i faceSize = ghostFluidWeights.size(axis);
		int totalFaceSamples = faceSize[0] * faceSize[1] * faceSize[2];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaceSamples, tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec3i face = ghostFluidWeights.grid(axis).unflatten(faceIndex);

				Vec3i backwardCell = faceToCell(face, axis, 0);
				Vec3i forwardCell = faceToCell(face, axis, 1);

				if (backwardCell[axis] < 0 || forwardCell[axis] >= surface.size()[axis])
					continue;
				else
				{
					float phiBackward = surface(backwardCell);
					float phiForward = surface(forwardCell);

					if (phiBackward < 0 || phiForward < 0)
						ghostFluidWeights(face, axis) = lengthFraction(phiBackward, phiForward);
				}
			}
		});
	}

	return ghostFluidWeights;
}

VectorGrid<float> computeCutCellWeights(const LevelSet& surface, bool invertWeights)
{
	VectorGrid<float> cutCellWeights(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	ScalarGrid<float> nodeSampledSurface(surface.xform(), surface.size(), 0, ScalarGridSettings::SampleType::NODE);

	{
		tbb::parallel_for(tbb::blocked_range<int>(0, nodeSampledSurface.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
			{
				Vec3i node = nodeSampledSurface.unflatten(sampleIndex);

				Vec3f worldNodePoint = nodeSampledSurface.indexToWorld(Vec3f(node));
				nodeSampledSurface(node) = surface.interp(worldNodePoint);
			}
		});
	}

	for (int faceAxis : {0, 1, 2})
	{
		Vec3i faceSize = cutCellWeights.size(faceAxis);
		int totalFaceSamples = faceSize[0] * faceSize[1] * faceSize[2];

		tbb::parallel_for(tbb::blocked_range<int>(0, totalFaceSamples, tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
			{
				Vec3i face = cutCellWeights.grid(faceAxis).unflatten(faceIndex);

				std::array<float, 4> nodePhis;

				for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
				{
					Vec3i node = faceToNodeCCW(face, faceAxis, nodeIndex);
					nodePhis[nodeIndex] = nodeSampledSurface(node);
				}

				float weight = fractionInside(nodePhis);
				weight = Utilities::clamp(weight, float(0), float(1));

				if (invertWeights)
					weight = 1. - weight;

				if (weight > 0)
					cutCellWeights(face, faceAxis) = weight;
			}
		});
	}

	return cutCellWeights;
}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.

void computeSupersampleVolumes(ScalarGrid<float>& volumes, const LevelSet& surface, int samples)
{
	assert(samples > 0);

	float dx = 1. / float(samples);
	float sampleVolume = Utilities::cube(dx);

	tbb::parallel_for(tbb::blocked_range<int>(0, volumes.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
	{
		for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
		{
			Vec3i sampleCoord = volumes.unflatten(sampleIndex);

			if (surface.interp(volumes.indexToWorld(Vec3f(sampleCoord))) > 2. * surface.dx())
				continue;

			Vec3f start = Vec3f(sampleCoord) - Vec3f(.5 - .5 * dx);
			Vec3f end = Vec3f(sampleCoord) + Vec3f(.5);

			Vec3f sample;
			float insideMaterialCount = 0;

			for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
				for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
					for (sample[2] = start[2]; sample[2] <= end[2]; sample[2] += dx)
					{
						Vec3f worldSample = volumes.indexToWorld(sample);

						if (surface.interp(worldSample) <= 0.)
							++insideMaterialCount;
					}

			volumes(sampleCoord) = insideMaterialCount * sampleVolume;
		}
	});
}

VectorGrid<float> computeSupersampledFaceVolumes(const LevelSet& surface, int samples)
{
	assert(samples > 0);

	VectorGrid<float> volumes(surface.xform(), surface.size(), 0, VectorGridSettings::SampleType::STAGGERED);

	float dx = 1. / float(samples);
	float sampleVolume = Utilities::cube(dx);

	for (int axis : {0, 1, 2})
	{
		tbb::parallel_for(tbb::blocked_range<int>(0, volumes.grid(axis).voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
		{
			for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
			{
				Vec3i sampleCoord = volumes.grid(axis).unflatten(sampleIndex);

				if (surface.interp(volumes.indexToWorld(Vec3f(sampleCoord), axis)) > 2. * surface.dx())
					continue;

				Vec3f start = Vec3f(sampleCoord) - Vec3f(.5 - .5 * dx);
				Vec3f end = Vec3f(sampleCoord) + Vec3R(.5);

				Vec3f sample;
				float insideMaterialCount = 0;

				for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
					for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
						for (sample[2] = start[2]; sample[2] <= end[2]; sample[2] += dx)
						{
							Vec3f worldSample = volumes.indexToWorld(sample, axis);

							if (surface.interp(worldSample) <= 0.)
								++insideMaterialCount;
						}

				volumes(sampleCoord, axis) = insideMaterialCount * sampleVolume;
			}
		});
	}

	return volumes;
}
}