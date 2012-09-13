/*
 * vec.hpp
 *
 *  Created on: November 26, 2011
 *      Author: Markus Doellinger
 */

#ifndef VEC_HPP_
#define VEC_HPP_

template<typename T>
class Vec {
public:
	Vec<T>(int n = 0);
	Vec<T>(int n, const T* const p);

	void init();

	T len() const;
	T lenlen() const;
	T dist(const Vec<T>& b) const;
	T distsqr(const Vec<T>& b) const;
	T distcos(const Vec<T>& b) const;
	T disttan(const Vec<T>& b) const;

	Vec<T> operator+(const Vec<T>& b) const;
	Vec<T> operator-(const Vec<T>& b) const;
	Vec<T> operator*(const T& t) const;

	T& operator[](const int i);
	const T& operator[](const int i) const;

	int N;
	T* v;
};

template<typename T>
inline
Vec<T>::Vec(int n)
{
	N = n;
	if (N == 0)
		v = NULL;
	else
		v = new T[N];
}

template<typename T>
inline
Vec<T>::Vec(int n, const T* const p)
{
	N = n;
	v = new T[N];
	memcpy(v, p, N * sizeof(T));
}

template<typename T>
inline
void Vec<T>::init(int n)
{
	if (N != 0)
		delete[] v;

	N = n;
	if (N == 0)
		v = NULL;
	else
		v = new T[N];
}

template<typename T>
inline
T Vec<T>::len() const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += v[i] * v[i];
	return sqrt(result);
}

template<typename T>
inline
T Vec<T>::lenlen() const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += v[i] * v[i];
	return result;
}

template<typename T>
inline
T Vec<T>::dist(const Vec<T>& b) const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += (v[i] - b[i]) * (v[i] - b[i]);
	return sqrt(result);
}

template<typename T>
inline
T Vec<T>::distsqr(const Vec<T>& b) const
{
	T result = 0.0f;
	for (int i = 0; i < N; ++i)
		result += (v[i] - b[i]) * (v[i] - b[i]);
	return result;
}

template<typename T>
inline
T Vec<T>::distcos(const Vec<T>& b) const
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

template<typename T>
inline
T Vec<T>::disttan(const Vec<T>& b) const
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


template<typename T>
inline
Vec<T> Vec<T>::operator+(const Vec<T>& b) const
{
	Vec<T> result;
	for (int i = 0; i < N; ++i)
		result[i] = v[i] + b[i];
	return result;
}

template<typename T>
inline
Vec<T> Vec<T>::operator-(const Vec<T>& b) const
{
	Vec<T> result;
	for (int i = 0; i < N; ++i)
		result[i] = v[i] - b[i];
	return result;
}



template<typename T>
inline
T& Vec<T>::operator[](const int i)
{
	return v[i];
}

template<typename T>
inline
const T& Vec<T>::operator[](const int i) const
{
	return v[i];
}


#endif /* VEC4_HPP_ */
