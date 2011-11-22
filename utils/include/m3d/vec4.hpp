/*
 * vec4.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef VEC4_HPP_
#define VEC4_HPP_

template<typename T>
class Vec4 {
public:
	Vec4<T>();
	Vec4<T>(const Vec4<int32_t>& v);
	Vec4<T>(const Vec4<float>& v);
	Vec4<T>(const Vec4<double>& v);
	Vec4<T>(const T& x, const T& y, const T& z, const T& w);
	Vec4<T>(const T* const v);
	Vec4<T>(const Vec3<T>& v, const T& w = (T)0);

#ifdef M3D_USE_OPENGL
	static Vec4<GLint> viewport();
#endif

	Vec4<T> normalized() const;
	void normalize();

	T len() const;
	T lenlen() const;

	Vec3<T> xyz() const;

	std::string str() const;
	void assign(std::string str);

	Vec4<T>& operator+=(const Vec4<T>&);
	Vec4<T>& operator-=(const Vec4<T>&);
	Vec4<T>& operator*=(const Mat4<T>& m);
	Vec4<T>& operator*=(const T& t);

	Vec4<T> operator-() const;

	Vec4<T> operator+(const Vec4<T>& v) const;
	Vec4<T> operator-(const Vec4<T>& v) const;
	Vec4<T> operator*(const T& t) const;
	Vec4<T> operator*(const Mat4<T>& m) const;
	T operator*(const Vec4<T>& v) const; ///< dot product
	Vec4<T> operator%(const Vec4<T>& v) const; ///< cross product

	bool operator==(const Vec4<T>& v) const;
    bool operator!=(const Vec4<T>& v) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	T x, y, z, w;
};

template<typename T>
inline
Vec4<T>::Vec4()
	: x((T)0), y((T)0), z((T)0), w((T)0)
{
}

template<typename T>
inline
Vec4<T>::Vec4(const Vec4<int32_t>& v)
	: x((T)v.x), y((T)v.y), z((T)v.z), w((T)v.w)
{
}

template<typename T>
inline
Vec4<T>::Vec4(const Vec4<float>& v)
	: x((T)v.x), y((T)v.y), z((T)v.z), w((T)v.w)
{
}

template<typename T>
inline
Vec4<T>::Vec4(const Vec4<double>& v)
	: x((T)v.x), y((T)v.y), z((T)v.z), w((T)v.w)
{
}

template<typename T>
inline
Vec4<T>::Vec4(const T& x, const T& y, const T& z, const T& w)
	: x(x), y(y), z(z), w(w)
{
}

template<typename T>
inline
Vec4<T>::Vec4(const Vec3<T>& v, const T& w)
	: x(v.x), y(v.y), z(v.z), w(w)
{
}

#ifdef M3D_USE_OPENGL
template<typename T>
inline
Vec4<GLint> Vec4<T>::viewport()
{
	Vec4<GLint> res;
	glGetIntegerv(GL_VIEWPORT, &res[0]);
	return res;
}
#endif


template<typename T>
inline
Vec4<T> Vec4<T>::normalized() const
{
	T tmp = 1 / sqrt(x*x + y*y + z*z + w*w);
	return Vec4<T>(x*tmp, y*tmp, z*tmp, w*tmp);
}

template<typename T>
inline
void Vec4<T>::normalize()
{
	T tmp = 1 / sqrt(x*x + y*y + z*z + w*w);
	x *= tmp;
	y *= tmp;
	z *= tmp;
	w *= tmp;
}

template<typename T>
inline
T Vec4<T>::len() const
{
	return sqrt(x*x + y*y + z*z + w*w);
}

template<typename T>
inline
T Vec4<T>::lenlen() const
{
	return x*x + y*y + z*z + w*w;
}

template<typename T>
inline
Vec3<T> Vec4<T>::xyz() const
{
	return Vec3<T>(x, y, z);
}

template<typename T>
inline
std::string Vec4<T>::str() const
{
	std::stringstream sst;
	sst << *this;
	return sst.str();
}

template<typename T>
inline
void Vec4<T>::assign(std::string str)
{
	std::stringstream sst;
	sst << str.c_str();
	sst.seekg(0, std::ios::beg);
	sst >> *this;
}

template<typename T>
inline
Vec4<T>& Vec4<T>::operator+=(const Vec4<T>& v)
{
	x += v.x;
	y += v.y;
	z += v.z;
	w += v.w;
	return *this;
}

template<typename T>
inline
Vec4<T>& Vec4<T>::operator-=(const Vec4<T>& v)
{
	x -= v.x;
	y -= v.y;
	z -= v.z;
	w -= v.w;
	return *this;
}

template<typename T>
inline
Vec4<T>& Vec4<T>::operator*=(const Mat4<T>& m)
{
	T _x = x*m._11 + y*m._21 + z*m._31 + w*m._41;
	T _y = x*m._12 + y*m._22 + z*m._32 + w*m._42;
	T _z = x*m._13 + y*m._23 + z*m._33 + w*m._43;
	T _w = x*m._14 + y*m._24 + z*m._34 + w*m._44;
	x = _x;
	y = _y;
	z = _z;
	w = _w;
	return *this;
}

template<typename T>
inline
Vec4<T>& Vec4<T>::operator*=(const T& t)
{
	x *= t;
	y *= t;
	z *= t;
	w *= t;
	return *this;
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator-() const
{
	return Vec4(-x, -y, -z, -w);
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator+(const Vec4<T>& v) const
{
	return Vec4<T>(x + v.x, y + v.y, z + v.z, w + v.w);
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator-(const Vec4<T>& v) const
{
	return Vec4<T>(x - v.x, y - v.y, z - v.z, w - v.w);
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator*(const T& t) const
{
	return Vec4<T>(x*t, y*t, z*t, w*t);
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator*(const Mat4<T>& m) const
{
	return Vec4<T>(x*m._11 + y*m._21 + z*m._31 + w*m._41,
	               x*m._12 + y*m._22 + z*m._32 + w*m._42,
	               x*m._13 + y*m._23 + z*m._33 + w*m._43,
	               x*m._14 + y*m._24 + z*m._34 + w*m._44);
}

template<typename T>
inline
T Vec4<T>::operator*(const Vec4<T>& v) const
{
	return (x*v.x + y*v.y + z*v.z + w*v.w);
}

template<typename T>
inline
Vec4<T> Vec4<T>::operator%(const Vec4<T>& v) const
{
	return Vec4<T>(y * v.z - z * v.y,
				   z * v.x - x * v.z,
				   x * v.y - y * v.x,
				   w);
}

template<typename T>
inline
bool Vec4<T>::operator==(const Vec4<T>& v) const
{
	return x == v.x && y == v.y && z == v.z && w == v.w;
}

template<typename T>
inline
bool Vec4<T>::operator!=(const Vec4<T>& v) const
{
	return x != v.x || y != v.y || z != v.z || w != v.w;
}


template<typename T>
inline
T& Vec4<T>::operator[](const int i)
{
	return *(&x + i);
}

template<typename T>
inline
const T& Vec4<T>::operator[](const int i) const
{
	return *(&x + i);
}

#endif /* VEC4_HPP_ */
