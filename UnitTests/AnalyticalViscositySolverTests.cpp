#include <Eigen/Sparse>

#include "gtest/gtest.h"

#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"

#include "ScalarGrid.h"
#include "Transform.h"
#include "Utilities.h"
#include "VectorGrid.h"

using namespace FluidSim3D;

class AnalyticalViscositySolver
{
    static constexpr int UNASSIGNED = -1;

public:
    AnalyticalViscositySolver(const Transform& xform, const Vec3i& size)
        : myXform(xform)
        , myVelocityIndex(myXform, size, Vec3i::Constant(UNASSIGNED), VectorGridSettings::SampleType::STAGGERED)
    {}

    // Returns the infinity-norm error of the numerical solution
    template<typename Initial, typename Solution, typename Viscosity>
    double solve(const Initial& initial, const Solution& solution, const Viscosity& viscosity, const double dt);

    Vec3d cellIndexToWorld(const Vec3d& index) const { return myXform.indexToWorld(index + Vec3d::Constant(.5)); }
    Vec3d edgeIndexToWorld(const Vec3d& index, int edgeAxis) const
    {
        Vec3d offset;
        if (edgeAxis == 0)
            offset = Vec3d(.5, 0., 0.);
        else if (edgeAxis == 1)
            offset = Vec3d(0., .5, 0.);
        else
            offset = Vec3d(0., 0., .5);

        return myXform.indexToWorld(index + offset);
    }

private:
    Transform myXform;
    int setVelocityIndices();
    VectorGrid<int> myVelocityIndex;
};

template<typename Initial, typename Solution, typename Viscosity>
double AnalyticalViscositySolver::solve(const Initial& initialFunction, const Solution& solutionFunction, const Viscosity& viscosityFunction, const double dt)
{
    int velocityDOFCount = setVelocityIndices();

    // Build reduced system.
    // (Note we don't need control volumes since the cells are the same size and there is no free surface).
    // (I - dt * mu * D^T K D) u(n+1) = u(n)

    std::vector<Eigen::Triplet<double>> sparseMatrixElements;

    VectorXd rhsVector = VectorXd::Zero(velocityDOFCount);

    double dx = myVelocityIndex.dx();
    double baseCoeff = dt / std::pow(dx, 2);

    tbb::enumerable_thread_specific<std::vector<Eigen::Triplet<double>>> parallelSparseElements;

    for (int faceAxis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myVelocityIndex.grid(faceAxis).voxelCount()), [&](const tbb::blocked_range<int>& range)
        {
            auto& localSparseElements = parallelSparseElements.local();

            for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
            {
                Vec3i face = myVelocityIndex.grid(faceAxis).unflatten(faceIndex);

                int velocityIndex = myVelocityIndex(face, faceAxis);

                if (velocityIndex >= 0)
                {
                    Vec3d facePosition = myVelocityIndex.indexToWorld(face.cast<double>(), faceAxis);

                    rhsVector(velocityIndex) += initialFunction(facePosition, faceAxis);

                    localSparseElements.emplace_back(velocityIndex, velocityIndex, 1);

                    // Build cell-centered stresses.
                    for (int divergenceDirection : {0, 1})
                    {
                        Vec3i cell = faceToCell(face, faceAxis, divergenceDirection);

                        Vec3d cellPosition = cellIndexToWorld(cell.cast<double>());
                        double cellCoeff = 2. * viscosityFunction(cellPosition) * baseCoeff;

                        double divSign = (divergenceDirection == 0) ? -1 : 1;

                        for (int gradientDirection : {0, 1})
                        {
                            Vec3i adjacentFace = cellToFace(cell, faceAxis, gradientDirection);

                            double gradSign = (gradientDirection == 0) ? -1 : 1;

                            int adjacentFaceIndex = myVelocityIndex(adjacentFace, faceAxis);
                            if (adjacentFaceIndex >= 0)
                                localSparseElements.emplace_back(velocityIndex, adjacentFaceIndex, -divSign * gradSign * cellCoeff);
                            else
                            {
                                Vec3d adjacentFacePosition = myVelocityIndex.indexToWorld(adjacentFace.cast<double>(), faceAxis);
                                rhsVector(velocityIndex) += divSign * gradSign * cellCoeff * solutionFunction(adjacentFacePosition, faceAxis);
                            }
                        }
                    }

                    for (int edgeAxis : {0, 1, 2})
                    {
                        if (edgeAxis == faceAxis) continue;

                        for (int divergenceDirection : {0, 1})
                        {
                            Vec3i edge = faceToEdge(face, faceAxis, edgeAxis, divergenceDirection);

                            Vec3d edgePosition = edgeIndexToWorld(edge.cast<double>(), edgeAxis);
                            double edgeCoeff = viscosityFunction(edgePosition) * baseCoeff;

                            double divSign = (divergenceDirection == 0) ? -1 : 1;

                            for (int gradientAxis : {0, 1, 2})
                            {
                                if (gradientAxis == edgeAxis) continue;

                                int gradientFaceAxis = 3 - gradientAxis - edgeAxis;
                                
                                for (int gradientDirection : {0, 1})
                                {
                                    double gradSign = (gradientDirection == 0) ? -1 : 1;

                                    Vec3i localGradientFace = edgeToFace(edge, edgeAxis, gradientFaceAxis, gradientDirection);

                                    // Check for out of bounds
                                    if (gradientDirection == 0 && localGradientFace[gradientAxis] < 0 || gradientDirection == 1 && localGradientFace[gradientAxis] >= myVelocityIndex.size(gradientFaceAxis)[gradientAxis])
                                    {
                                        Vec3d gradientFacePosition = myVelocityIndex.indexToWorld(localGradientFace.cast<double>(), gradientFaceAxis);
                                        rhsVector(velocityIndex) += divSign * gradSign * edgeCoeff * solutionFunction(gradientFacePosition, gradientFaceAxis);
                                    }
                                    else
                                    {
                                        int gradientFaceIndex = myVelocityIndex(localGradientFace, gradientFaceAxis);

                                        if (gradientFaceIndex >= 0)
                                            localSparseElements.emplace_back(velocityIndex, gradientFaceIndex, -divSign * gradSign * edgeCoeff);
                                        else
                                        {
                                            Vec3d gradientFacePosition = myVelocityIndex.indexToWorld(localGradientFace.cast<double>(), gradientFaceAxis);
                                            rhsVector(velocityIndex) += divSign * gradSign * edgeCoeff * solutionFunction(gradientFacePosition, gradientFaceAxis);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }            
        });

        mergeLocalThreadVectors(sparseMatrixElements, parallelSparseElements);
    }

    Eigen::SparseMatrix<double> sparseMatrix(velocityDOFCount, velocityDOFCount);
    sparseMatrix.setFromTriplets(sparseMatrixElements.begin(), sparseMatrixElements.end());

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper | Eigen::Lower> solver;
    solver.compute(sparseMatrix);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

    VectorXd solutionVector = solver.solve(rhsVector);

    if (solver.info() != Eigen::Success)
    {
        return -1;
    }

    double error = 0;

    for (int axis : {0, 1})
    {
        double localError = tbb::parallel_reduce(tbb::blocked_range<int>(0, myVelocityIndex.grid(axis).voxelCount()), double(0),
            [&](const tbb::blocked_range<int>& range, double error) -> double {
                for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                {
                    Vec3i face = myVelocityIndex.grid(axis).unflatten(faceIndex);

                    int velocityIndex = myVelocityIndex(face, axis);

                    if (velocityIndex >= 0)
                    {
                        Vec3d facePosition = myVelocityIndex.indexToWorld(face.cast<double>(), axis);
                        double localError = fabs(solutionVector(velocityIndex) - solutionFunction(facePosition, axis));

                        error = std::max(error, localError);
                    }
                }

                return error;
            },
            [](double a, double b) -> double {
                return std::max(a, b);
            });

        error = std::max(error, localError);
    }

    return error;
}

int AnalyticalViscositySolver::setVelocityIndices()
{
    // Loop over each face. If it's not along the boundary
    // then include it into the system.

    int index = 0;

    for (int axis : {0, 1, 2})
    {
        Vec3i size = myVelocityIndex.size(axis);

        forEachVoxelRange(Vec3i::Zero(), size, [&](const Vec3i& face)
        {
            // Faces along the boundary are removed from the simulation
            if (face[axis] > 0 && face[axis] < size[axis] - 1)
                myVelocityIndex(face, axis) = index++;
        });
    }
    // Returning index gives the number of velocity positions required for a linear system
    return index;
}

TEST(ANALYTICAL_VISCOSITY_TEST, CONVERGENCE_TEST)
{
    double dt = 1.;

    auto solution = [](const Vec3d& pos, int)
    {
        return std::sin(pos[0]) * std::sin(pos[1]) * std::sin(pos[2]);
    };
    auto viscosity = [](const Vec3d& pos) { return 1.; };

    auto initial = [&](const Vec3d& pos, int axis)
    {
        double x = pos[0];
        double y = pos[1];
        double z = pos[2];
      
        if (axis == 0)
            return (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2. - (std::cos(x) * std::cos(z) * std::sin(y)) / 2. - (std::cos(x) * std::cos(y) * std::sin(z)) / 2. + (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2.;
        else if (axis == 1)
            return (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2. - (std::cos(y) * std::cos(z) * std::sin(x)) / 2. - (std::cos(x) * std::cos(y) * std::sin(z)) / 2. + (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2.;
        else
            return (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2. - (std::cos(y) * std::cos(z) * std::sin(x)) / 2. - (std::cos(x) * std::cos(z) * std::sin(y)) / 2. + (3. * std::sin(x) * std::sin(y) * std::sin(z)) / 2.;
    };

    const int startGrid = 8;
    const int endGrid = startGrid * int(pow(2, 4));

    std::vector<double> errors;
    for (int gridSize = startGrid; gridSize < endGrid; gridSize *= 2)
    {
        const double dx = 1. / double(gridSize);
        Vec3d origin = Vec3d::Zero();
        Vec3i size = Vec3i::Constant(startGrid);
        Transform xform(dx, origin);

        AnalyticalViscositySolver solver(xform, size);
        double error = solver.solve(initial, solution, viscosity, dt);

        errors.push_back(error);
        EXPECT_GT(error, 0.);
    }

    for (int errorIndex = 1; errorIndex < errors.size(); ++errorIndex)
    {
        double errorRatio = errors[errorIndex - 1] / errors[errorIndex];
        EXPECT_GT(errorRatio, 4.);
    }
}
