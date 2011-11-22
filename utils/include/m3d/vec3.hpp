/*
 * vec3.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef VEC3_HPP_
#define VEC3_HPP_

template<typename T>
class Vec3 {
public:
	Vec3<T>();
	Vec3<T>(const Vec3<float>& v);
	Vec3<T>(const Vec3<double>& v);
	Vec3<T>(const T& x, const T& y, const T& z);
	Vec3<T>(const T* const v);

	/**
	 * Returns the x axis.
	 *
	 * @return The vector (1, 0, 0)
	 */
	static Vec3<T> xAxis();

	/**
	 * Returns the y axis.
	 *
	 * @return The vector (0, 1, 0)
	 */
	static Vec3<T> yAxis();

	/**
	 * Returns the z axis.
	 *
	 * @return The vector (0, 0, 1)
	 */
	static Vec3<T> zAxis();

	Vec3<T> normalized() const; ///< Returns the vector in a normalized form
	T normalize(); ///< normalizes the vector

	T len() const; ///< returns the length of the vector
	T lenlen() const; ///< returns the squared length of the vector

	Vec2<T> xz() const;

	std::string str() const;
	void assign(std::string str);

	Vec3<T>& operator+=(const Vec3<T>& v);
	Vec3<T>& operator-=(const Vec3<T>& v);
	Vec3<T>& operator*=(const Mat4<T>& m);
	Vec3<T>& operator*=(const T& t);
	Vec3<T>& operator/=(const T& t);

	Vec3<T> operator-() const;

	Vec3<T> operator+(const Vec3<T>& v) const;
	Vec3<T> operator-(const Vec3<T>& v) const;
	Vec3<T> operator*(const T& t) const;
	Vec3<T> operator/(const T& t) const;
	Vec3<T> operator*(const Mat4<T>& m) const; ///< matrix multiplication
	Vec3<T> operator%(const Mat4<T>& m) const; ///< matrix multiplication without transformation
	T operator*(const Vec3<T>& v) const; ///< dot product
	Vec3<T> operator%(const Vec3<T>& v) const; //< cross product

    bool operator!=(const Vec3<T>& v) const;
	bool operator==(const Vec3<T>& v) const;
	bool operator<(const Vec3<T>& v) const;
	bool operator<=(const Vec3<T>& v) const;
	bool operator>=(const Vec3<T>& v) const;
	bool operator>(const Vec3<T>& v) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	T x, y, z;
};

template<typename T>
inline
Vec3<T>::Vec3()
	: x((T)0), y((T)0), z((T)0)
{
}

template<typename T>
inline
Vec3<T>::Vec3(const Vec3<float>& v)
	: x((T)v.x), y((T)v.y), z((T)v.z)
{
}

template<typename T>
inline
Vec3<T>::Vec3(const Vec3<double>& v)
	: x((T)v.x), y((T)v.y), z((T)v.z)
{
}

template<typename T>
inline
Vec3<T>::Vec3(const T& x, const T& y, const T& z)
	: x(x), y(y), z(z)
{
}

template<typename T>
inline
Vec3<T>::Vec3(const T* v)
{
	x = v[0];
	y = v[1];
	z = v[2];
}


template<typename T>
inline
Vec3<T> Vec3<T>::xAxis()
{
	return Vec3<T>(1.0f, 0.0f, 0.0f);
}

template<typename T>
inline
Vec3<T> Vec3<T>::yAxis()
{
	return Vec3<T>(0.0f, 1.0f, 0.0f);
}

template<typename T>
inline
Vec3<T> Vec3<T>::zAxis()
{
	return Vec3<T>(0.0f, 0.0f, 1.0f);
}


template<typename T>
inline
Vec3<T> Vec3<T>::normalized() const
{
	T tmp = 1 / sqrt(x*x + y*y + z*z);
	return Vec3<T>(x*tmp, y*tmp, z*tmp);
}

template<typename T>
inline
T Vec3<T>::normalize()
{
	T res = sqrt(x*x + y*y + z*z);
	T tmp = 1 / res;
	x *= tmp;
	y *= tmp;
	z *= tmp;
	return res;
}

template<typename T>
inline
T Vec3<T>::len() const
{
	return sqrt(x*x + y*y + z*z);
}

template<typename T>
inline
T Vec3<T>::lenlen() const
{
	return (x*x + y*y + z*z);
}

template<typename T>
inline
Vec2<T> Vec3<T>::xz() const
{
	return Vec2<T>(x, z);
}

template<typename T>
inline
std::string Vec3<T>::str() const
{
	std::stringstream sst;
	sst << *this;
	return sst.str();
}

template<typename T>
inline
void Vec3<T>::assign(std::string str)
{
	std::stringstream sst;
	sst << str.c_str();
	sst.seekg(0, std::ios::beg);
	sst >> *this;
}

template<typename T>
inline
Vec3<T>& Vec3<T>::operator+=(const Vec3<T>& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	return *this;
}

template<typename T>
inline
Vec3<T>& Vec3<T>::operator-=(const Vec3<T>& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	return *this;
}

template<typename T>
inline
Vec3<T>& Vec3<T>::operator*=(const Mat4<T>& m)
{
	T _x = x*m._11 + y*m._21 + z*m._31 + m._41;
	T _y = x*m._12 + y*m._22 + z*m._32 + m._42;
	T _z = x*m._13 + y*m._23 + z*m._33 + m._43;
	x = _x; y = _y; z = _z;
	return *this;
}

template<typename T>
inline
Vec3<T>& Vec3<T>::operator*=(const T& t)
{
	x *= t;
	y *= t;
	z *= t;
	return *this;
}

template<typename T>
inline
Vec3<T>& Vec3<T>::operator/=(const T& t)
{
	x /= t;
	y /= t;
	z /= t;
	return *this;
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator-() const
{
	return Vec3(-x, -y, -z);
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator+(const Vec3<T>& v) const
{
	return Vec3<T>(x + v.x, y + v.y, z + v.z);
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator-(const Vec3<T>& v) const
{
	return Vec3<T>(x - v.x, y - v.y, z - v.z);
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator*(const T& t) const
{
	return Vec3<T>(x*t, y*t, z*t);
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator/(const T& t) const
{
	return Vec3<T>(x/t, y/t, z/t);
}

template<typename T> inline Vec3<T> Vec3<T>::operator*(const Mat4<T>& m) const
{
	return Vec3<T>(x*m._11 + y*m._21 + z*m._31 + m._41,
				   x*m._12 + y*m._22 + z*m._32 + m._42,
				   x*m._13 + y*m._23 + z*m._33 + m._43);
}

template<typename T> inline Vec3<T> Vec3<T>::operator%(const Mat4<T>& m) const
{
	return Vec3<T>(x*m._11 + y*m._21 + z*m._31,
				   x*m._12 + y*m._22 + z*m._32,
	               x*m._13 + y*m._23 + z*m._33);
}

template<typename T>
inline
T Vec3<T>::operator*(const Vec3<T>& v) const
{
	return x*v.x + y*v.y + z*v.z;
}

template<typename T>
inline
Vec3<T> Vec3<T>::operator%(const Vec3<T>& v) const
{
	return Vec3<T>(y * v.z - z * v.y,
				   z * v.x - x * v.z,
				   x * v.y - y * v.x);
}

template<typename T>
inline
bool Vec3<T>::operator!=(const Vec3<T>& v) const
{
	return x != v.x || y != v.y || z != v.z;
}

template<typename T>
inline
bool Vec3<T>::operator<(const Vec3<T>& v) const
{
	return x < v.x && y < v.y && z < v.z;
}

template<typename T>
inline
bool Vec3<T>::operator<=(const Vec3<T>& v) const
{
	return x <= v.x && y <= v.y && z <= v.z;
}

template<typename T>
inline
bool Vec3<T>::operator==(const Vec3<T>& v) const
{
	return x == v.x && y == v.y && z == v.z;
}

template<typename T>
inline
bool Vec3<T>::operator>=(const Vec3<T>& v) const
{
	return x >= v.x && y >= v.y && z >= v.z;
}

template<typename T>
inline
bool Vec3<T>::operator>(const Vec3<T>& v) const
{
	return x > v.x && y > v.y && z > v.z;
}

template<typename T>
inline
T& Vec3<T>::operator[](const int i)
{
	return *(&x + i);
}

template<typename T>
inline
const T& Vec3<T>::operator[](const int i) const
{
	return *(&x + i);
}

#endif /* VEC3_HPP_ */
