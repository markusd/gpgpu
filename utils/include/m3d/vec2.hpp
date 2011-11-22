/*
 * vec2.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef VEC2_HPP_
#define VEC2_HPP_

/**
 * A two dimensional vector with a parameterized component type.
 */
template<typename T>
class Vec2 {
public:
	Vec2<T>(); ///< Initializes all components to 0
	Vec2<T>(const Vec2<float>& v); ///< Copies the vector from a 2D float vector
	Vec2<T>(const Vec2<double>& v); ///< Copies the vector from a 2D double vector
	Vec2<T>(const T& x, const T& y); ///< Constructs a vector from x and y values
	Vec2<T>(const T* const v);

	Vec2<T> normalized() const; ///< Returns the vector in a normalized form
	void normalize(); ///< normalizes the vector

	T len() const; ///< returns the length of the vector
	T lenlen() const; ///< returns the squared length of the vector

	Vec3<T> xz3(T _y = (T)0) const; ///< returns a 3D vector using x and z and the given y value

	Vec2<T>& operator+=(const Vec2<T>&);
	Vec2<T>& operator-=(const Vec2<T>&);
	Vec2<T>& operator*=(const T& t);

	Vec2<T> operator-() const;

	Vec2<T> operator+(const Vec2<T>& v) const;
	Vec2<T> operator-(const Vec2<T>& v) const;
	Vec2<T> operator*(const T& t) const;
	T operator*(const Vec2<T>& v) const; ///< dot product
	T operator%(const Vec2<T>& v) const; ///< cross product

    bool operator!=(const Vec2<T>& v) const;
	bool operator==(const Vec2<T>& v) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	T x, y;
};

template<typename T>
inline
Vec2<T>::Vec2()
	: x((T)0), y((T)0)
{
}

template<typename T>
inline
Vec2<T>::Vec2(const Vec2<float>& v)
	: x((T)v.x), y((T)v.y)
{
}

template<typename T>
inline
Vec2<T>::Vec2(const Vec2<double>& v)
	: x((T)v.x), y((T)v.y)
{
}

template<typename T>
inline
Vec2<T>::Vec2(const T& x, const T& y)
	: x(x), y(y)
{
}

template<typename T>
inline
Vec2<T>::Vec2(const T* const v)
{
	x = v[0];
	y = v[1];
}


template<typename T>
inline
Vec2<T> Vec2<T>::normalized() const
{
	T tmp = 1 / sqrt(x*x + y*y);
	return Vec2<T>(x*tmp, y*tmp);
}

template<typename T>
inline
void Vec2<T>::normalize()
{
	T tmp = 1 / sqrt(x*x + y*y);
	x *= tmp;
	y *= tmp;
}

template<typename T>
inline
T Vec2<T>::len() const
{
	return sqrt(x*x + y*y);
}

template<typename T>
inline
T Vec2<T>::lenlen() const
{
	return x*x + y*y;
}

template<typename T>
inline
Vec3<T> Vec2<T>::xz3(T _y) const
{
	return Vec3<T>(x, _y, y);
}

template<typename T>
inline
Vec2<T>& Vec2<T>::operator+=(const Vec2<T>& v)
{
	x += v.x;
	y += v.y;
	return *this;
}

template<typename T>
inline
Vec2<T>& Vec2<T>::operator-=(const Vec2<T>& v)
{
	x -= v.x;
	y -= v.y;
	return *this;
}

template<typename T>
inline
Vec2<T>& Vec2<T>::operator*=(const T& t)
{
	x *= t;
	y *= t;
	return *this;
}

template<typename T>
inline
Vec2<T> Vec2<T>::operator-() const
{
	return Vec2<T>(-x, -y);
}

template<typename T>
inline
Vec2<T> Vec2<T>::operator+(const Vec2<T>& v) const
{
	return Vec2<T>(x + v.x, y + v.y);
}

template<typename T>
inline
Vec2<T> Vec2<T>::operator-(const Vec2<T>& v) const
{
	return Vec2<T>(x - v.x, y - v.y);
}

template<typename T>
inline
Vec2<T> Vec2<T>::operator*(const T& t) const
{
	return Vec2<T>(x*t, y*t);
}

template<typename T>
inline
T Vec2<T>::operator*(const Vec2<T>& v) const
{
	return x*v.x + y*v.y;
}

template<typename T>
inline
T Vec2<T>::operator%(const Vec2<T>& v) const
{
	return x*v.y - y*v.x;
}

template<typename T>
inline
bool Vec2<T>::operator==(const Vec2<T>& v) const
{
	return x == v.x && y == v.y;
}

template<typename T>
inline
bool Vec2<T>::operator!=(const Vec2<T>& v) const
{
	return x != v.x || y != v.y;
}



template<typename T>
inline
T& Vec2<T>::operator[](const int i)
{
	return *(&x + i);
}

template<typename T>
inline
const T& Vec2<T>::operator[](const int i) const
{
	return *(&x + i);
}


#endif /* VEC2_HPP_ */
