/**
 * @author Markus Doellinger, Robert Waury
 * @date May 23, 2011
 * @file util/clock.cpp
 */

#include <util/clock.hpp>
#ifdef _WIN32
	#include <windows.h>
#else
	#include <sys/time.h>
#endif

namespace util {

Clock::Clock()
{
	reset();
}

Clock::~Clock()
{
}

double Clock::sysTime() const
{
#ifdef _WIN32
	static LARGE_INTEGER freq, start;
	double time;

	// returns counts per second
	QueryPerformanceFrequency(&freq);

	// returns counts since boot
	QueryPerformanceCounter(&start);

	time = (double)start.QuadPart/(double)freq.QuadPart;
	
	return time;
#else
	timeval time = { 0, 0 };
	gettimeofday(&time, 0);
	return time.tv_sec + time.tv_usec / 1000000.0;
#endif
}

void Clock::reset()
{
	m_reference = sysTime();
}

float Clock::get() const
{
	return sysTime() - m_reference;
}

}
