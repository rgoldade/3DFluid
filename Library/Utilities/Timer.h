#ifndef FLUIDSIM3D_TIMER_H
#define FLUIDSIM3D_TIMER_H

#include <chrono>

///////////////////////////////////
//
// Timer.h
// Ryan Goldade 2017
//
////////////////////////////////////

namespace FluidSim3D
{
class Timer
{
public:
    Timer() : myStart(std::chrono::high_resolution_clock::now()) {}

    double stop()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - myStart;
        return elapsed.count();
    }

    void reset()
    {
        myStart = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::high_resolution_clock::time_point myStart;
};

}
#endif
