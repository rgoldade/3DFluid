#include "ComputeWeights.h"

#include <array>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

namespace FluidSim3D
{

// Helper functions to compute face area fractions for the cut-cell weights.
// Taken from Christopher Batty's source code.
void rotateFaceValues(std::array<double, 4>& phiNodes)
{
    double firstNodePhi = phiNodes[0];
    for (int i = 0; i < 3; ++i) phiNodes[i] = phiNodes[i + 1];
    phiNodes[3] = firstNodePhi;
}

// Given four signed distance values (square corners), determine what fraction of the square is "inside" a surface.
// This boils down to simple geometric operations that compute the area of implicit triangles over the face.
// This is largely borrowed from Dr. Christopher Batty's sample code. (https://cs.uwaterloo.ca/~c2batty/).

// Node layout is CCW.
// 3 -- 2
// |    |
// 0 -- 1

double fractionInside(std::array<double, 4>& phiNodes)
{
    // double phiBottomLeft, double phiBottomRight, double phiTopLeft, double phiTopRight)
    int insideCount = (phiNodes[0] < 0. ? 1 : 0) + (phiNodes[1] < 0. ? 1 : 0) + (phiNodes[2] < 0. ? 1 : 0) +
                      (phiNodes[3] < 0. ? 1 : 0);

    if (insideCount == 4)
        return 1.;
    else if (insideCount == 3)
    {
        // Rotate until the positive value is in the first position
        while (phiNodes[0] < 0.) rotateFaceValues(phiNodes);

        // Work out the area of the exterior triangle
        double side0 = 1. - lengthFraction(phiNodes[0], phiNodes[3]);
        double side1 = 1. - lengthFraction(phiNodes[0], phiNodes[1]);
        return 1. - 0.5 * side0 * side1;
    }
    else if (insideCount == 2)
    {
        // Rotate until a negative value is in the first position, and the next negative is in either slot 1 or 2.
        while (phiNodes[0] >= 0. || !(phiNodes[1] < 0. || phiNodes[2] < 0.)) rotateFaceValues(phiNodes);

        if (phiNodes[1] < 0)  // The matching signs are adjacent
        {
            double sideLeft = lengthFraction(phiNodes[0], phiNodes[3]);
            double sideRight = lengthFraction(phiNodes[1], phiNodes[2]);
            return 0.5 * (sideLeft + sideRight);
        }
        else  // The matching signs are diagonally opposite
        {
            // Determine the centre point's sign to disambiguate this case
            double phiMiddle = 0.25 * (phiNodes[0] + phiNodes[1] + phiNodes[2] + phiNodes[3]);
            if (phiMiddle < 0.)
            {
                double area = 0.;

                // First triangle (top left)
                double side1 = 1. - lengthFraction(phiNodes[0], phiNodes[3]);
                double side3 = 1. - lengthFraction(phiNodes[2], phiNodes[3]);

                area += 0.5 * side1 * side3;

                // Second triangle (top right)
                double side2 = 1. - lengthFraction(phiNodes[2], phiNodes[1]);
                double side0 = 1. - lengthFraction(phiNodes[0], phiNodes[1]);

                area += 0.5 * side0 * side2;

                return 1. - area;
            }
            else
            {
                double area = 0.;

                // First triangle (bottom left)
                double side0 = lengthFraction(phiNodes[0], phiNodes[1]);
                double side1 = lengthFraction(phiNodes[0], phiNodes[3]);
                area += 0.5 * side0 * side1;

                // Second triangle (top right)
                double side2 = lengthFraction(phiNodes[2], phiNodes[1]);
                double side3 = lengthFraction(phiNodes[2], phiNodes[3]);
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
        double side0 = lengthFraction(phiNodes[0], phiNodes[3]);
        double side1 = lengthFraction(phiNodes[0], phiNodes[1]);
        return 0.5 * side0 * side1;
    }

    return 0.;
}

VectorGrid<double> computeGhostFluidWeights(const LevelSet& surface)
{
    VectorGrid<double> ghostFluidWeights(surface.xform(), surface.size(), Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, ghostFluidWeights.grid(axis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
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
                    double phiBackward = surface(backwardCell);
                    double phiForward = surface(forwardCell);

                    if (phiBackward <= 0 || phiForward <= 0)
                        ghostFluidWeights(face, axis) = lengthFraction(phiBackward, phiForward);
                }
            }
        });
    }

    return ghostFluidWeights;
}

VectorGrid<double> computeCutCellWeights(const LevelSet& surface, bool invertWeights)
{
    VectorGrid<double> cutCellWeights(surface.xform(), surface.size(), Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

    ScalarGrid<double> nodeSampledSurface(surface.xform(), surface.size(), 0, ScalarGridSettings::SampleType::NODE);

    {
        tbb::parallel_for(tbb::blocked_range<int>(0, nodeSampledSurface.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
            {
                Vec3i node = nodeSampledSurface.unflatten(sampleIndex);

                Vec3d worldNodePoint = nodeSampledSurface.indexToWorld(node.cast<double>());
                nodeSampledSurface(node) = surface.triLerp(worldNodePoint);
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

                std::array<double, 4> nodePhis;

                for (int nodeIndex = 0; nodeIndex < 4; ++nodeIndex)
                {
                    Vec3i node = faceToNodeCCW(face, faceAxis, nodeIndex);
                    nodePhis[nodeIndex] = nodeSampledSurface(node);
                }

                double weight = fractionInside(nodePhis);
                weight = std::clamp(weight, double(0), double(1));

                if (invertWeights) weight = 1. - weight;

                if (weight > 0) cutCellWeights(face, faceAxis) = weight;
            }
        });
    }

    return cutCellWeights;
}

// There is no assumption about grid alignment for this method because
// we're computing weights for centers, faces, nodes, etc. that each
// have their internal index space cell offsets. We can't make any
// easy general assumptions about indices between grids anymore.

void computeSupersampleVolumes(ScalarGrid<double>& volumes, const LevelSet& surface, int samples)
{
    assert(samples > 0);

    double dx = 1. / double(samples);
    double sampleVolume = std::pow(dx, 3);

    tbb::parallel_for(tbb::blocked_range<int>(0, volumes.voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
    {
        for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
        {
            Vec3i sampleCoord = volumes.unflatten(sampleIndex);

            double sdf = surface.triLerp(volumes.indexToWorld(sampleCoord.cast<double>()));
            if (sdf > 2. * surface.dx())
                continue;
            
            if (sdf < -2. * surface.dx())
            {
                volumes(sampleCoord) = 1;
                continue;
            }

            Vec3d start = sampleCoord.cast<double>() - Vec3d::Constant(.5 - .5 * dx);
            Vec3d end = sampleCoord.cast<double>() + Vec3d::Constant(.5);

            Vec3d sample;
            double insideMaterialCount = 0;

            for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
                for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
                    for (sample[2] = start[2]; sample[2] <= end[2]; sample[2] += dx)
                    {
                        Vec3d worldSamplePoint = volumes.indexToWorld(sample);

                        if (surface.triLerp(worldSamplePoint) <= 0.) ++insideMaterialCount;
                    }

            if (insideMaterialCount > 0) volumes(sampleCoord) = insideMaterialCount * sampleVolume;
        }
    });
}

VectorGrid<double> computeSupersampledFaceVolumes(const LevelSet& surface, int samples)
{
    assert(samples > 0);

    VectorGrid<double> volumes(surface.xform(), surface.size(), Vec3d::Zero(), VectorGridSettings::SampleType::STAGGERED);

    double dx = 1. / double(samples);

    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, volumes.grid(axis).voxelCount(), tbbHeavyGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int sampleIndex = range.begin(); sampleIndex != range.end(); ++sampleIndex)
            {
                Vec3i sampleCoord = volumes.grid(axis).unflatten(sampleIndex);

                double sdf = surface.triLerp(volumes.indexToWorld(sampleCoord.cast<double>(), axis));
                if (sdf > 2. * surface.dx())
                    continue;
                if (sdf < -2 * surface.dx())
                {
                    volumes(sampleCoord, axis) = 1;
                    continue;
                }

                Vec3d start = sampleCoord.cast<double>() - Vec3d::Constant(.5 - .5 * dx);
                Vec3d end = sampleCoord.cast<double>() + Vec3d::Constant(.5);

                Vec3d sample;
                double insideMaterialCount = 0;
                int sampleCount = 0;

                for (sample[0] = start[0]; sample[0] <= end[0]; sample[0] += dx)
                    for (sample[1] = start[1]; sample[1] <= end[1]; sample[1] += dx)
                        for (sample[2] = start[2]; sample[2] <= end[2]; sample[2] += dx)
                        {
                            Vec3d worldSample = volumes.indexToWorld(sample, axis);

                            if (surface.triLerp(worldSample) <= 0.) ++insideMaterialCount;

                            ++sampleCount;
                        }

                if (insideMaterialCount > 0) volumes(sampleCoord, axis) = insideMaterialCount / double(sampleCount);
            }
        });
    }

    return volumes;
}

}