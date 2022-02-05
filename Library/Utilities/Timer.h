#ifndef FLUIDSIM3D_TIMER_H
#define FLUIDSIM3D_TIMER_H

#ifdef _MSC_VER
#include <Windows.h>
#else
#include <sys/time.h>
#endif

#include "Utilities.h"

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
    Timer()
    {
#ifdef _MSC_VER
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
#else
        struct timezone tz;
        gettimeofday(&m_start, &tz);
#endif
    }

    double stop()
    {
#ifdef _MSC_VER
        QueryPerformanceCounter(&EndingTime);
        ElapsedMicroseconds.QuadPart = EndingTime.QuadPart - StartingTime.QuadPart;

        return double(ElapsedMicroseconds.QuadPart) / double(Frequency.QuadPart);
#else

        struct timezone tz;
        gettimeofday(&m_end, &tz);

        return (m_end.tv_sec + m_end.tv_usec * 1E-6) - (m_start.tv_sec + m_start.tv_usec * 1E-6);
#endif
    }

    void reset()
    {
#ifdef _MSC_VER
        QueryPerformanceFrequency(&Frequency);
        QueryPerformanceCounter(&StartingTime);
#else
        struct timezone tz;
        gettimeofday(&m_start, &tz);
#endif
    }

private:
#ifdef _MSC_VER
    LARGE_INTEGER StartingTime, EndingTime, ElapsedMicroseconds;
    LARGE_INTEGER Frequency;
#else
    struct timeval m_start, m_end;
#endif
};

}
#endif