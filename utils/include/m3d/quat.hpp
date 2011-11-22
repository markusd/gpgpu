/*
 * quat.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef QUAT_HPP_
#define QUAT_HPP_

template<typename T>
class Quat {
public:
	Quat<T>();
	Quat<T>(const Quat<float>& q);
	Quat<T>(const Quat<double>& q);
	Quat<T>(const T& a, const T& b, const T& c, const T& d);
	Quat<T>(const T* const q);
	Quat<T>(const Vec3<T>& p);
	Quat<T>(const Vec3<T>& axis, const T& angle);

	T norm() const;
	Mat4<T> mat4() const;
	Vec3<T> point() const;
	T angle() const;
	Vec3<T> axis() const;
	Vec3<T> rotate(const Vec3<T> v) const;
	void conjugate();
	Quat<T> conjugated() const;

	Quat<T>& operator+=(const Quat<T>& q);
	Quat<T>& operator-=(const Quat<T>& q);
	Quat<T>& operator*=(const Quat<T>& q);
	Quat<T>& operator*=(const T& t);

	Quat<T> operator-() const;

	Quat<T> operator+(const Quat<T>& q) const;
	Quat<T> operator-(const Quat<T>& q) const;
	Quat<T> operator*(const T& t) const;
	Quat<T> operator*(const Quat<T>& q) const;
	bool operator==(const Quat<T>& q) const;
    bool operator!=(const Quat<T>& q) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	T a, b, c, d;
};

template<typename T>
inline
Quat<T>::Quat()
	: a((T)0), b((T)0), c((T)0), d((T)0)
{
}

template<typename T>
inline
Quat<T>::Quat(const Quat<float>& q)
	: a((T)q.a), b((T)q.b), c((T)q.d), d((T)q.d)
{
}

template<typename T>
inline
Quat<T>::Quat(const Quat<double>& q)
	: a((T)q.a), b((T)q.b), c((T)q.d), d((T)q.d)
{
}

template<typename T>
inline
Quat<T>::Quat(const T& a, const T& b, const T& c, const T& d)
	: a(a), b(b), c(c), d(d)
{
}

template<typename T>
inline
Quat<T>::Quat(const Vec3<T>& p)
	: a((T)0), b(p.x), c(p.y), d(p.z)
{
}

template<typename T>
inline
Quat<T>::Quat(const Vec3<T>& axis, const T& angle)
{
	T _half = angle/2.0f;
	T _sin = sin(_half);
	a = cos(_half);
	b = _sin * axis.x;
	c = _sin * axis.y;
	d = _sin * axis.z;
}

template<typename T>
inline
Quat<T>& Quat<T>::operator+=(const Quat<T>& q)
{
	a += q.a;
	b += q.b;
	c += q.c;
	d += q.d;
	return *this;
}

template<typename T>
inline
Quat<T>& Quat<T>::operator-=(const Quat<T>& q)
{
	a -= q.a;
	b -= q.b;
	c -= q.c;
	d -= q.d;
	return *this;
}

template<typename T>
inline
Quat<T>& Quat<T>::operator*=(const Quat<T>& q)
{
	T tA = (a*q.a - b*q.b - c*q.c - d*q.d);
	T tB = (b*q.a + a*q.b - d*q.c + c*q.d);
	T tC = (c*q.a + d*q.b + a*q.c - b*q.d);
	T tD = (d*q.a - c*q.b + b*q.c + a*q.d);
	a = tA;
	b = tB;
	c = tC;
	d = tD;
	return *this;
}

template<typename T>
inline
Quat<T>& Quat<T>::operator*=(const T& t)
{
	a *= t;
	b *= t;
	c *= t;
	d *= t;
	return *this;
}

template<typename T>
inline
Quat<T> Quat<T>::operator-() const
{
	return Quat(-a, -b, -c, -d);
}

template<typename T>
inline
Quat<T> Quat<T>::operator+(const Quat<T>& q) const
{
	return Quat<T>(a + q.a, b + q.b, c + q.c, d + q.d);
}

template<typename T>
inline
Quat<T> Quat<T>::operator-(const Quat<T>& q) const
{
	return Quat<T>(a - q.a, b - q.b, c - q.c, d - q.d);
}

template<typename T>
inline
Quat<T> Quat<T>::operator*(const T& t) const
{
	return Quat<T>(a*t, b*t, c*t, d*t);
}

template<typename T>
inline
Quat<T> Quat<T>::operator*(const Quat<T>& q) const
{
	return Quat<T>((a*q.a - b*q.b - c*q.c - d*q.d),
				   (b*q.a + a*q.b - d*q.c + c*q.d),
				   (c*q.a + d*q.b + a*q.c - b*q.d),
				   (d*q.a - c*q.b + b*q.c + a*q.d));
}

template<typename T>
inline
bool Quat<T>::operator==(const Quat<T>& q) const
{
	return a == q.a && b == q.b && c == q.c && d == q.d;
}

template<typename T>
inline
bool Quat<T>::operator!=(const Quat<T>& q) const
{
	return a != q.a || b != q.b || c != q.c || d != q.d;
}

template<typename T>
inline
T Quat<T>::norm() const
{
	return sqrt(a*a + b*b + c*c + d*d);
}


template<typename T>
inline
Mat4<T> Quat<T>::mat4() const
{
	Vec4<T> x((a*a + b*b - c*c - d*d), 2*(c*b+a*d)            , 2*(d*b-a*c)            , 0.0f);
	Vec4<T> y(2*(b*c-a*d)            , (a*a - b*b + c*c - d*d), 2*(d*c+a*b)            , 0.0f);
	Vec4<T> z(2*(b*d+a*c)            , 2*(c*d-a*b)            , (a*a - b*b - c*c + d*d), 0.0f);
	Vec4<T> w(0.0f                   , 0.0f                   , 0.0f                   , 1.0f);
	return Mat4<T>(x, y, z, w);
}

template<typename T>
inline
Vec3<T> Quat<T>::point() const
{
	return Vec3<T>(b, c, d);
}

template<typename T>
inline
T Quat<T>::angle() const
{
	double _tmp = a;
	if (_tmp >  1.0) _tmp =  1.0;
	if (_tmp < -1.0) _tmp = -1.0;
	return acos(_tmp) * 2.0;
}

template<typename T>
inline
Vec3<T> Quat<T>::axis() const
{
	double _tmp = a;
	if (_tmp >  1.0) _tmp =  1.0;
	if (_tmp < -1.0) _tmp = -1.0;
	double _half = acos(_tmp);
	if (_half < -EPSILON || _half > EPSILON) {
		double _isin = 1.0 / sin(_half);
		return Vec3<T>(b * _isin, c * _isin, d * _isin);
	}
	else {
		return Vec3<T>();
	}
}

template<typename T>
inline
Vec3<T> Quat<T>::rotate(const Vec3<T> v) const
{
	return v * mat4();
}

template<typename T>
inline
void Quat<T>::conjugate()
{
	b = -b;
	c = -c;
	d = -d;
}

template<typename T>
inline
Quat<T> Quat<T>::conjugated() const
{
	return Quat<T>(a, -b, -c, -d);
}

template<typename T>
inline
T& Quat<T>::operator[](const int i)
{
	return *(&a + i);
}

template<typename T>
inline
const T& Quat<T>::operator[](const int i) const
{
	return *(&a + i);
}


#endif /* QUAT_HPP_ */
