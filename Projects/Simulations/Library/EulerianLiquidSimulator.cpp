#include "EulerianLiquidSimulator.h"

#include <iostream>

#include "ComputeWeights.h"
#include "ExtrapolateField.h"
#include "PressureProjection.h"
#include "Timer.h"
#include "ViscositySolver.h"

void EulerianLiquidSimulator::drawGrid(Renderer& renderer, bool onlyDrawNarrowBand) const
{
    myLiquidSurface.drawGrid(renderer, onlyDrawNarrowBand);
}

void EulerianLiquidSimulator::drawLiquidSurface(Renderer& renderer) const
{
    TriMesh liquidSurfaceMesh = myLiquidSurface.buildMesh();
    liquidSurfaceMesh.drawMesh(renderer, true /* render tri faces */, Vec3f(.5), false /* don't render tri normals */,
                               Vec3f(0), false /* don't render tri vertices */, Vec3f(0), true /* render tri edges */,
                               Vec3f(0, 1, 0));
}

void EulerianLiquidSimulator::drawLiquidVelocity(Renderer& renderer, Axis planeAxis, float planePosition,
                                                 float length) const
{
    myLiquidVelocity.drawSamplePointVectors(renderer, planeAxis, planePosition, Vec3f(0),
                                            myLiquidVelocity.dx() * length);
}

void EulerianLiquidSimulator::drawSolidSurface(Renderer& renderer) const
{
    TriMesh solidSurfaceMesh = mySolidSurface.buildMesh();
    solidSurfaceMesh.drawMesh(renderer, false /* don't render tri faces */, Vec3f(0),
                              false /* don't render tri normals */, Vec3f(0), false /* don't render tri vertices */,
                              Vec3f(0), true /* render tri edges */, Vec3f(.25));
}

void EulerianLiquidSimulator::drawSolidVelocity(Renderer& renderer, Axis planeAxis, float planePosition,
                                                float length) const
{
    mySolidVelocity.drawSamplePointVectors(renderer, planeAxis, planePosition, Vec3f(0, 1, 0),
                                           mySolidVelocity.dx() * length);
}

void EulerianLiquidSimulator::setSolidSurface(const LevelSet& solidSurface)
{
    assert(solidSurface.isBackgroundNegative());

    TriMesh localMesh = solidSurface.buildMesh();

    mySolidSurface.setBackgroundNegative();
    mySolidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquidSimulator::setSolidVelocity(const VectorGrid<float>& solidVelocity)
{
    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, solidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
                          [&](tbb::blocked_range<int>& range) {
                              for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                              {
                                  Vec3i face = mySolidVelocity.grid(axis).unflatten(faceIndex);

                                  Vec3f facePosition = mySolidVelocity.indexToWorld(Vec3f(face), axis);
                                  mySolidVelocity(face, axis) = solidVelocity.interp(facePosition, axis);
                              }
                          });
    }
}

void EulerianLiquidSimulator::setLiquidSurface(const LevelSet& surface)
{
    TriMesh localMesh = surface.buildMesh();
    myLiquidSurface.initFromMesh(localMesh, false);
}

void EulerianLiquidSimulator::setLiquidVelocity(const VectorGrid<float>& velocity)
{
    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
                          [&](tbb::blocked_range<int>& range) {
                              for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                              {
                                  Vec3i face = myLiquidVelocity.grid(axis).unflatten(faceIndex);

                                  Vec3f facePosition = myLiquidVelocity.indexToWorld(Vec3f(face), axis);
                                  myLiquidVelocity(face, axis) = velocity.interp(facePosition, axis);
                              }
                          });
    }
}

void EulerianLiquidSimulator::unionLiquidSurface(const LevelSet& addedLiquidSurface)
{
    // Need to zero out velocity in this added region as it could get extrapolated values
    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(
            tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
            [&](tbb::blocked_range<int>& range) {
                for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                {
                    Vec3i face = myLiquidVelocity.grid(axis).unflatten(faceIndex);

                    Vec3f facePosition = myLiquidVelocity.indexToWorld(Vec3f(face), axis);
                    if (addedLiquidSurface.interp(facePosition) <= 0. && myLiquidSurface.interp(facePosition) > 0.)
                        myLiquidVelocity(face, axis) = 0;
                }
            });
    }

    // Combine surfaces
    myLiquidSurface.unionSurface(addedLiquidSurface);
}

template <typename ForceSampler>
void EulerianLiquidSimulator::addForce(float dt, const ForceSampler& force)
{
    for (int axis : {0, 1, 2})
    {
        tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidVelocity.grid(axis).voxelCount(), tbbLightGrainSize),
                          [&](tbb::blocked_range<int>& range) {
                              for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                              {
                                  Vec3i face = myLiquidVelocity.grid(axis).unflatten(faceIndex);

                                  Vec3f facePosition = myLiquidVelocity.indexToWorld(Vec3f(face), axis);
                                  myLiquidVelocity(face, axis) += dt * force(facePosition, axis);
                              }
                          });
    }
}

void EulerianLiquidSimulator::addForce(float dt, const Vec3f& force)
{
    addForce(dt, [&](Vec3f, int axis) { return force[axis]; });
}

void EulerianLiquidSimulator::advectOldPressure(float dt)
{
    auto velocityFunc = [&](float, const Vec3f& pos) { return myLiquidVelocity.interp(pos); };

    ScalarGrid<float> tempPressure(myOldPressure.xform(), myOldPressure.size());

    advectField(dt, tempPressure, myOldPressure, velocityFunc, IntegrationOrder::RK3);

    std::swap(myOldPressure, tempPressure);
}

void EulerianLiquidSimulator::advectLiquidSurface(float dt, IntegrationOrder integrator)
{
    auto velocityFunc = [&](float, const Vec3f& point) { return myLiquidVelocity.interp(point); };

    TriMesh localMesh = myLiquidSurface.buildMesh();
    localMesh.advectMesh(dt, velocityFunc, integrator);
    assert(localMesh.unitTestMesh());

    myLiquidSurface.initFromMesh(localMesh, false);

    // Remove solid regions from liquid surface
    tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidSurface.voxelCount(), tbbLightGrainSize),
                      [&](tbb::blocked_range<int>& range) {
                          for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                          {
                              Vec3i cell = myLiquidSurface.unflatten(cellIndex);
                              myLiquidSurface(cell) = std::max(myLiquidSurface(cell), -mySolidSurface(cell));
                          }
                      });

    myLiquidSurface.reinitMesh();
}

void EulerianLiquidSimulator::advectViscosity(float dt, IntegrationOrder integrator)
{
    auto velocityFunc = [&](float, const Vec3f& point) { return myLiquidVelocity.interp(point); };

    ScalarGrid<float> tempViscosity(myViscosity.xform(), myViscosity.size());

    advectField(dt, tempViscosity, myViscosity, velocityFunc, integrator);

    std::swap(tempViscosity, myViscosity);
}

void EulerianLiquidSimulator::advectLiquidVelocity(float dt, IntegrationOrder integrator)
{
    auto velocityFunc = [&](float, const Vec3f& point) { return myLiquidVelocity.interp(point); };

    VectorGrid<float> tempVelocity(myLiquidSurface.xform(), myLiquidSurface.size(),
                                   VectorGridSettings::SampleType::STAGGERED);

    for (int axis : {0, 1, 2})
        advectField(dt, tempVelocity.grid(axis), myLiquidVelocity.grid(axis), velocityFunc, integrator);

    std::swap(myLiquidVelocity, tempVelocity);
}

void EulerianLiquidSimulator::runTimestep(float dt)
{
    std::cout << "\nStarting simulation loop\n" << std::endl;

    Timer simTimer;

    LevelSet extrapolatedSurface = myLiquidSurface;

    float dx = extrapolatedSurface.dx();

    tbb::parallel_for(tbb::blocked_range<int>(0, myLiquidSurface.voxelCount(), tbbLightGrainSize),
                      [&](tbb::blocked_range<int>& range) {
                          for (int cellIndex = range.begin(); cellIndex != range.end(); ++cellIndex)
                          {
                              Vec3i cell = myLiquidSurface.unflatten(cellIndex);

                              if (mySolidSurface(cell) <= 0) extrapolatedSurface(cell) -= dx;
                          }
                      });

    extrapolatedSurface.reinitMesh();

    std::cout << "  Extrapolate into solids: " << simTimer.stop() << "s" << std::endl;
    simTimer.reset();

    VectorGrid<float> cutCellWeights = computeCutCellWeights(mySolidSurface, true);
    VectorGrid<float> ghostFluidWeights = computeGhostFluidWeights(extrapolatedSurface);

    std::cout << "  Compute weights: " << simTimer.stop() << "s" << std::endl;
    simTimer.reset();

    // Initialize and call pressure projection
    PressureProjection projectDivergence(extrapolatedSurface, cutCellWeights, ghostFluidWeights, mySolidVelocity);

    projectDivergence.setInitialGuess(myOldPressure);
    projectDivergence.project(myLiquidVelocity);

    myOldPressure = projectDivergence.getPressureGrid();

    const VectorGrid<VisitedCellLabels>& validFaces = projectDivergence.getValidFaces();

    assert(validFaces.isGridMatched(myLiquidVelocity));

    if (myDoSolveViscosity)
    {
        std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
        simTimer.reset();

        ViscositySolver(dt, myLiquidSurface, myLiquidVelocity, mySolidSurface, mySolidVelocity, myViscosity);

        std::cout << "  Solve for viscosity: " << simTimer.stop() << "s" << std::endl;
        simTimer.reset();

        projectDivergence.disableInitialGuess();
        projectDivergence.project(myLiquidVelocity);

        std::cout << "  Solve for pressure after viscosity: " << simTimer.stop() << "s" << std::endl;
        simTimer.reset();
    }
    else
    {
        std::cout << "  Solve for pressure: " << simTimer.stop() << "s" << std::endl;
        simTimer.reset();
    }

    // Extrapolate velocity
    for (int axis : {0, 1, 2})
    {
        // Zero out non-valid faces
        tbb::parallel_for(tbb::blocked_range<int>(0, validFaces.grid(axis).voxelCount(), tbbLightGrainSize),
                          [&](const tbb::blocked_range<int>& range) {
                              for (int faceIndex = range.begin(); faceIndex != range.end(); ++faceIndex)
                              {
                                  Vec3i face = validFaces.grid(axis).unflatten(faceIndex);

                                  if (validFaces(face, axis) != VisitedCellLabels::FINISHED_CELL)
                                      myLiquidVelocity(face, axis) = 0;
                              }
                          });

        extrapolateField(myLiquidVelocity.grid(axis), validFaces.grid(axis), 1.5 * myCFL);
    }

    std::cout << "  Extrapolate velocity: " << simTimer.stop() << "s" << std::endl;
    simTimer.reset();

    advectOldPressure(dt);
    advectLiquidSurface(dt, IntegrationOrder::RK3);

    if (myDoSolveViscosity) advectViscosity(dt, IntegrationOrder::FORWARDEULER);

    advectLiquidVelocity(dt, IntegrationOrder::RK3);

    std::cout << "  Advect simulation: " << simTimer.stop() << "s" << std::endl;
}