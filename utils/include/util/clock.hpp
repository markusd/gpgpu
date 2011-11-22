/**
 * @author Markus Doellinger, Robert Waury
 * @date May 23, 2011
 * @file util/clock.hpp
 */

#ifndef CLOCK_HPP_
#define CLOCK_HPP_

namespace util
{

/**
 * Simple timer class that returns the time elapsed since a reference
 * value in seconds.
 */
class Clock {
protected:
	/**
	 * The reference value. All time computations are relative to this
	 * value.
	 */
	double m_reference;

	/**
	 * Returns the current system time. This is used to calculate the time
	 * elapsed since a reference point.
	 *
	 * @return Returns the current system time
	 */
	double sysTime() const;
public:
	Clock();
	virtual ~Clock();

	/**
	 * Sets the point of reference of the clock to the current time. All
	 * calls to get() will return the time elapsed since a call to this
	 * method.
	 */
	void reset();

	/**
	 * Returns the time elapsed since the last call to reset() in seconds.
	 *
	 * @return The time elapsed in seconds
	 */
	float get() const;
};

}

#endif /* CLOCK_HPP_ */
