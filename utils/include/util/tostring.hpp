/**
 * @author Robert Waury
 * @date Jun 9, 2011
 * @file util/tostring.hpp
 */

#ifndef TOSTRING_HPP_
#define TOSTRING_HPP_

#include <iostream>
#include <sstream>
#include <string>

namespace util {

/**
 *	Performs conversion to string
 *
 *	@param	value	parameter to convert
 *	@return pointer to converted string
*/
template<typename T>
char* toString(T value)
{
	std::stringstream sst;
	sst << value;
	sst.seekg(0, std::ios::beg);
	return strdup(sst.str().c_str());
}

template<typename T>
int toInt(T value)
{
	std::stringstream sst;
	sst << value;
	sst.seekg(0, std::ios::beg);
	int result;
	sst >> result;
	return result;
}

}

#endif /* TOSTRING_HPP_ */
