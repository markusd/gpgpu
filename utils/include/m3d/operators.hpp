/*
 * operators.hpp
 *
 *  Created on: May 14, 2011
 *      Author: Markus Doellinger
 */

#ifndef OPERATORS_HPP_
#define OPERATORS_HPP_

template<typename T>
inline
std::ostream& operator<< (std::ostream& out, const Vec2<T>& v)
{
	out.setf(std::ios::fixed, std::ios::floatfield);
	out.precision(10);
	return out << v.x << ", " << v.y;
}

template<typename T>
inline
std::ostream& operator<< (std::ostream& out, const Vec3<T>& v)
{
	out.setf(std::ios::fixed, std::ios::floatfield);
	out.precision(10);
	return out << v.x << ", " << v.y << ", " << v.z;
}

template<typename T>
inline
std::istream& operator>> (std::istream& is, Vec3<T>& v)
{
	is.exceptions(std::istream::failbit | std::istream::badbit);
	is >> v.x; is.ignore(2);
	is >> v.y; is.ignore(2);
	return is >> v.z;
}

template<typename T>
inline
std::ostream& operator<< (std::ostream& out, const Vec4<T>& v)
{
	out.setf(std::ios::fixed, std::ios::floatfield);
	out.precision(10);
	return out << v.x << ", " << v.y << ", " << v.z << ", " << v.w;
}

template<typename T>
inline
std::istream& operator>> (std::istream& is, Vec4<T>& v)
{
	is.exceptions(std::istream::failbit | std::istream::badbit);
	is >> v.x; is.ignore(2);
	is >> v.y; is.ignore(2);
	is >> v.z; is.ignore(2);
	return is >> v.w;
}

template<typename T>
inline
std::ostream& operator<< (std::ostream& out, const Mat4<T>& m)
{
	out.setf(std::ios::fixed, std::ios::floatfield);
	out.precision(10);
	return out << m._11 << ", " << m._12 << ", " << m._13 << ", " << m._14 << "; "
	           << m._21 << ", " << m._22 << ", " << m._23 << ", " << m._24 << "; "
	           << m._31 << ", " << m._32 << ", " << m._33 << ", " << m._34 << "; "
	           << m._41 << ", " << m._42 << ", " << m._43 << ", " << m._44;
}

template<typename T>
inline
std::istream& operator>> (std::istream& is, Mat4<T>& m)
{
	is.exceptions(std::istream::failbit | std::istream::badbit);
	is >> m._11; is.ignore(2); is >> m._12; is.ignore(2); is >> m._13; is.ignore(2); is >> m._14; is.ignore(2);
	is >> m._21; is.ignore(2); is >> m._22; is.ignore(2); is >> m._23; is.ignore(2); is >> m._24; is.ignore(2);
	is >> m._31; is.ignore(2); is >> m._32; is.ignore(2); is >> m._33; is.ignore(2); is >> m._34; is.ignore(2);
	is >> m._41; is.ignore(2); is >> m._42; is.ignore(2); is >> m._43; is.ignore(2); is >> m._44;
	return is;
}

template<typename T>
inline
std::ostream& operator<< (std::ostream& out, const Quat<T>& q)
{
	out.setf(std::ios::fixed, std::ios::floatfield);
	out.precision(10);
	return out << q.a << ", " << q.b << ", " << q.c << ", " << q.d;
}


#endif /* OPERATORS_HPP_ */
