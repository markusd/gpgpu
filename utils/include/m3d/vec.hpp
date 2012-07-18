/*
 * vec.hpp
 *
 *  Created on: November 26, 2011
 *      Author: Markus Doellinger
 */

#ifndef VEC_HPP_
#define VEC_HPP_

template<int N, typename T>
class Vec {
public:
	Vec<N,T>();
	Vec<N,T>(const T* const p);

	T len() const;
	T lenlen() const;
	T dist(const Vec<N,T>& b) const;
	T distsqr(const Vec<N,T>& b) const;
	T distcos(const Vec<N,T>& b) const;
	T disttan(const Vec<N,T>& b) const;

	Vec<N,T> operator+(const Vec<N,T>& b) const;
	Vec<N,T> operator-(const Vec<N,T>& b) const;
	Vec<N,T> operator*(const T& t) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	T v[N];
};

template<int N, typename T>
inline
Vec<N,T>::Vec()
{
}

template<int N, typename T>
inline
Vec<N,T>::Vec(const T* const p)
{
	memcpy(v, p, n * sizeof(T));
}


template<int N, typename T>
inline
T Vec<N,T>::len() const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += v[i] * v[i];
	return sqrt(result);
}

template<int N, typename T>
inline
T Vec<N,T>::lenlen() const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += v[i] * v[i];
	return result;
}

template<int N, typename T>
inline
T Vec<N,T>::dist(const Vec<N,T>& b) const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += (v[i] - b[i]) * (v[i] - b[i]);
	return sqrt(result);
}

template<int N, typename T>
inline
T Vec<N,T>::distsqr(const Vec<N,T>& b) const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += (v[i] - b[i]) * (v[i] - b[i]);
	return result;
}

template<int N, typename T>
inline
T Vec<N,T>::distcos(const Vec<N,T>& b) const
{
	T result = 0.0f;
	T lena = 0.0f;
	T lenb = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += v[i] * b[i];
		lena += v[i] * v[i];
		lenb += b[i] * b[i];
	}

	return 1.0f - result / (sqrt(lena) * sqrt(lenb));
}

template<int N, typename T>
inline
T Vec<N,T>::disttan(const Vec<N,T>& b) const
{
	T result = 0.0f;
	T lena = 0.0f;
	T lenb = 0.0f;

	for (int i = 0; i < DIM; ++i) {
		result += v[i] * b[i];
		lena += v[i] * v[i];
		lenb += b[i] * b[i];
	}

	return 1.0f - result / (lena + lenb - result);
}


template<int N, typename T>
inline
Vec<N,T> Vec<N,T>::operator+(const Vec<N,T>& b) const
{
	Vec<N,T> result;
	for (int i = 0; i < N; ++i)
		result[i] = v[i] + b[i];
	return result;
}

template<int N, typename T>
inline
Vec<N,T> Vec<N,T>::operator-(const Vec<N,T>& b) const
{
	Vec<N,T> result;
	for (int i = 0; i < N; ++i)
		result[i] = v[i] - b[i];
	return result;
}



template<int N, typename T>
inline
T& Vec<N,T>::operator[](const int i)
{
	return v[i];
}

template<int N, typename T>
inline
const T& Vec<N,T>::operator[](const int i) const
{
	return v[i];
}


#endif /* VEC4_HPP_ */
