#include "AnalyticalPoissonSolver.h"

void AnalyticalPoissonSolver::drawGrid(const std::string& label) const { myPoissonGrid.drawGrid(label); }

void AnalyticalPoissonSolver::drawValues(const std::string& label) const { myPoissonGrid.drawSupersampledValuesVolume(label); }
