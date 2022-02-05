#include "ViscositySolver.h"

#include <iostream>

#include <Eigen/Sparse>

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "ComputeWeights.h"
#include "LevelSet.h"

namespace FluidSim3D
{

void ViscositySolver(double dt,
                    const LevelSet& surface,
                    VectorGrid<double>& velocity,
                    const LevelSet& solidSurface,
                    const VectorGrid<double>& solidVelocity,
                    const ScalarGrid<double>& viscosity)
{
    // For efficiency sake, this should only take in velocity on a staggered grid
    // that matches the center sampled surface and collision
    assert(surface.isGridMatched(solidSurface));
    assert(surface.isGridMatched(viscosity));
    assert(velocity.isGridMatched(solidVelocity));

    for (int axis : {0, 1, 2})
    {
        Vec3i faceSize = velocity.size(axis);
        --faceSize[axis];

        assert(faceSize == surface.size());
    }

    int volumeSamples = 3;

    ScalarGrid<double> centerVolumes(surface.xform(), surface.size(), 0, ScalarGridSettings::SampleType::CENTER);
    computeSupersampleVolumes(centerVolumes, surface, 3);

    VectorGrid<double> edgeVolumes(surface.xform(), surface.size(), Vec3d::Zero(), VectorGridSettings::SampleType::EDGE);
    for (int axis : {0, 1, 2}) computeSupersampleVolumes(edgeVolumes.grid(axis), surface, 3);

    VectorGrid<double> faceVolumes = computeSupersampledFaceVolumes(surface, 3);

    enum class MaterialLabels
    {
        SOLID_FACE,
        LIQUID_FACE,
        AIR_FACE
    };

    VectorGrid<MaterialLabels> materialFaceLabels(surface.xform(), surface.size(), Vec3t<MaterialLabels>::Constant(MaterialLabels::AIR_FACE),
                                                  VectorGridSettings::SampleType::STAGGERED);

    // Set material labels for each grid face. We assume faces along the simulation boundary
    // are solid.

    for (int faceAxis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
            {
                Vec3i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);

                if (face[faceAxis] == 0 || face[faceAxis] == materialFaceLabels.size(faceAxis)[faceAxis] - 1)
                    continue;

                bool isFaceInSolve = false;

                for (int direction : {0, 1})
                {
                    Vec3i cell = faceToCell(face, faceAxis, direction);
                    if (centerVolumes(cell) > 0) isFaceInSolve = true;
                }

                if (!isFaceInSolve)
                {
                    for (int edgeAxis : {0, 1, 2})
                    {
                        if (edgeAxis == faceAxis) continue;

                        for (int direction : {0, 1})
                        {
                            Vec3i edge = faceToEdge(face, faceAxis, edgeAxis, direction);

                            if (edgeVolumes(edge, edgeAxis) > 0) isFaceInSolve = true;
                        }
                    }
                }

                if (isFaceInSolve)
                {
                    if (solidSurface.triLerp(materialFaceLabels.indexToWorld(face.cast<double>(), faceAxis)) <= 0.)
                        materialFaceLabels(face, faceAxis) = MaterialLabels::SOLID_FACE;
                    else
                        materialFaceLabels(face, faceAxis) = MaterialLabels::LIQUID_FACE;
                }
            }
        });
    }

    int liquidDOFCount = 0;

    constexpr int UNLABELLED_CELL = -1;

    VectorGrid<int> liquidFaceIndices(surface.xform(), surface.size(), Vec3t<int>::Constant(UNLABELLED_CELL),
                                      VectorGridSettings::SampleType::STAGGERED);

    for (int axis : {0, 1, 2})
    {
        forEachVoxelRange(Vec3i::Zero(), materialFaceLabels.size(axis), [&](const Vec3i& face)
        {
            if (materialFaceLabels(face, axis) == MaterialLabels::LIQUID_FACE)
                liquidFaceIndices(face, axis) = liquidDOFCount++;
        });
    }

    double discreteScalar = dt / std::pow(surface.dx(), 2);

    // Pre-scale all the control volumes with coefficients to reduce
    // redundant operations when building the linear system.

    tbb::parallel_for(tbb::blocked_range<int>(0, centerVolumes.voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
    {
        for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
        {
            Vec3i cell = centerVolumes.unflatten(cellIndex);

            if (centerVolumes(cell) > 0) centerVolumes(cell) *= 2. * discreteScalar * viscosity(cell);
        }
    });

    for (int edgeAxis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, edgeVolumes.grid(edgeAxis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int edgeIndex = range.begin(); edgeIndex != range.end(); ++edgeIndex)
            {
                Vec3i edge = edgeVolumes.grid(edgeAxis).unflatten(edgeIndex);

                if (edgeVolumes(edge, edgeAxis) > 0)
                    edgeVolumes(edge, edgeAxis) *= discreteScalar * viscosity.triLerp(edgeVolumes.indexToWorld(edge.cast<double>(), edgeAxis));
            }
        });
    }

    std::vector<Eigen::Triplet<double>> sparseElements;
    VectorXd initialGuessVector = VectorXd::Zero(liquidDOFCount);
    VectorXd rhsVector = VectorXd::Zero(liquidDOFCount);

    {
        tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>> parallelSparseElements;

        for (int faceAxis : {0, 1, 2})
        {
            tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
            {
                auto& localSparseElements = parallelSparseElements.local();

                for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                {
                    Vec3i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);

                    int liquidFaceIndex = liquidFaceIndices(face, faceAxis);

                    if (liquidFaceIndex >= 0)
                    {
                        assert(materialFaceLabels(face, faceAxis) == MaterialLabels::LIQUID_FACE);

                        // Use old velocity as an initial guess since we're solving for a new
                        // velocity field with viscous forces applied to the old velocity field.
                        initialGuessVector(liquidFaceIndex) = velocity(face, faceAxis);

                        // Build RHS with volume weights
                        double localFaceVolume = faceVolumes(face, faceAxis);

                        rhsVector(liquidFaceIndex) = localFaceVolume * velocity(face, faceAxis);

                        // Add volume weight to diagonal
                        double diagonal = localFaceVolume;

                        // Build cell centered stress terms
                        for (int divergenceDirection : {0, 1})
                        {
                            Vec3i cell = faceToCell(face, faceAxis, divergenceDirection);

                            assert(cell[faceAxis] >= 0 && cell[faceAxis] < centerVolumes.size()[faceAxis]);

                            double divergenceSign = (divergenceDirection == 0) ? -1 : 1;

                            if (centerVolumes(cell) > 0)
                            {
                                for (int gradientDirection : {0, 1})
                                {
                                    Vec3i adjacentFace = cellToFace(cell, faceAxis, gradientDirection);

                                    double gradientSign = (gradientDirection == 0) ? -1. : 1.;

                                    double coefficient = divergenceSign * gradientSign * centerVolumes(cell);

                                    int adjacentFaceIndex = liquidFaceIndices(adjacentFace, faceAxis);
                                    if (adjacentFaceIndex >= 0)
                                    {
                                        if (adjacentFaceIndex == liquidFaceIndex)
                                            diagonal -= coefficient;
                                        else
                                            localSparseElements.emplace_back(liquidFaceIndex, adjacentFaceIndex, -coefficient);
                                    }
                                    else if (materialFaceLabels(adjacentFace, faceAxis) == MaterialLabels::SOLID_FACE)
                                        rhsVector(liquidFaceIndex) += coefficient * solidVelocity(adjacentFace, faceAxis);
                                    else
                                        assert(materialFaceLabels(adjacentFace, faceAxis) == MaterialLabels::AIR_FACE);
                                }
                            }
                        }

                        for (int edgeAxis : {0, 1, 2})
                        {
                            if (edgeAxis == faceAxis) continue;

                            for (int divergenceDirection : {0, 1})
                            {
                                Vec3i edge = faceToEdge(face, faceAxis, edgeAxis, divergenceDirection);

                                if (edgeVolumes(edge, edgeAxis) > 0)
                                {
                                    double divergenceSign = (divergenceDirection == 0) ? -1 : 1;

                                    for (int gradientAxis : {0, 1, 2})
                                    {
                                        if (gradientAxis == edgeAxis) continue;

                                        int gradientFaceAxis = 3 - gradientAxis - edgeAxis;

                                        for (int gradientDirection : {0, 1})
                                        {
                                            double gradientSign = (gradientDirection == 0) ? -1 : 1;

                                            Vec3i localGradientFace = edgeToFace(edge, edgeAxis, gradientFaceAxis, gradientDirection);

                                            int gradientFaceIndex = liquidFaceIndices(localGradientFace, gradientFaceAxis);

                                            double coefficient = divergenceSign * gradientSign * edgeVolumes(edge, edgeAxis);
                                            if (gradientFaceIndex >= 0)
                                            {
                                                if (gradientFaceIndex == liquidFaceIndex)
                                                    diagonal -= coefficient;
                                                else
                                                    localSparseElements.emplace_back(liquidFaceIndex, gradientFaceIndex, -coefficient);
                                            }
                                            else if (materialFaceLabels(localGradientFace, gradientFaceAxis) == MaterialLabels::SOLID_FACE)
                                                rhsVector(liquidFaceIndex) += coefficient * solidVelocity(localGradientFace, gradientFaceAxis);
                                            else
                                                assert(materialFaceLabels(localGradientFace, gradientFaceAxis) == MaterialLabels::AIR_FACE);
                                        }
                                    }
                                }
                            }
                        }

                        localSparseElements.emplace_back(liquidFaceIndex, liquidFaceIndex, diagonal);
                    }
                    else
                        assert(materialFaceLabels(face, faceAxis) != MaterialLabels::LIQUID_FACE);
                }
            });
        }

        mergeLocalThreadVectors(sparseElements, parallelSparseElements);
    }

    SparseMatrix sparseMatrix(liquidDOFCount, liquidDOFCount);
    sparseMatrix.setFromTriplets(sparseElements.begin(), sparseElements.end());

    Eigen::ConjugateGradient<SparseMatrix, Eigen::Upper | Eigen::Lower> solver;
    solver.compute(sparseMatrix);
    solver.setTolerance(1E-3);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "   Solver failed to build" << std::endl;
        return;
    }

    VectorXd solutionVector = solver.solveWithGuess(rhsVector, initialGuessVector);

    if (solver.info() != Eigen::Success)
    {
        std::cout << "   Solver failed to converge" << std::endl;
        return;
    }
    else
    {
        std::cout << "    Solver iterations:     " << solver.iterations() << std::endl;
        std::cout << "    Solver error: " << solver.error() << std::endl;
    }

    for (int faceAxis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, materialFaceLabels.grid(faceAxis).voxelCount(), tbbLightGrainSize), [&](const tbb::blocked_range<int>& range)
        {
            for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
            {
                Vec3i face = materialFaceLabels.grid(faceAxis).unflatten(faceIndex);

                int liquidFaceIndex = liquidFaceIndices(face, faceAxis);
                if (liquidFaceIndex >= 0)
                {
                    assert(materialFaceLabels(face, faceAxis) == MaterialLabels::LIQUID_FACE);
                    velocity(face, faceAxis) = solutionVector(liquidFaceIndex);
                }
            }
        });
    }
}

}