#include <iostream>

#include "AnalyticalPoissonSolver.h"

#include "Utilities.h"

using namespace FluidSim3D;

int main(int, char**)
{
    auto rhs = [](const Vec3d& pos) -> double { return 3. * std::exp(-pos[0] - pos[1] - pos[2]); };

    auto solution = [](const Vec3d& pos) -> double { return std::exp(-pos[0] - pos[1] - pos[2]); };

	int baseGrid = 32;
    int maxBaseGrid = baseGrid * int(pow(2, 4));

    for (; baseGrid < maxBaseGrid; baseGrid *= 2)
    {
        double dx = PI / float(baseGrid);
        Vec3d origin = Vec3d::Zero();
        Vec3i size = Vec3i::Constant(int(round(PI / dx)));
        Transform xform(dx, origin);

        AnalyticalPoissonSolver solver(xform, size);
        double error = solver.solve(rhs, solution);

        std::cout << "L-infinity error at " << baseGrid << "^3: " << error << std::endl;
    }
}