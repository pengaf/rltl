#pragma once
#include "utility.h"
#include <type_traits>

BEGIN_RLTL_IMPL

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class Array;

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class ArrayLike
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	static const size_t t_dim = 1;
	static const size_t t_size = t_size_0;

	static Array<size_t, t_dim> shape()
	{
		Array<size_t, t_dim> shp;
		shp[0] = t_size_0;
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
	}
};


template<typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
class ArrayLike<Element_t, t_size_0, t_size_1, t_sizes...>
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	static const size_t t_sub_size = ArrayLike<Element_t, t_size_1, t_sizes...>::t_size;
	static const size_t t_sub_dims = ArrayLike<Element_t, t_size_1, t_sizes...>::t_dim;
	static const size_t t_dim = 1 + t_sub_dims;
	static const size_t t_size = t_size_0 * t_sub_size;

	static Array<size_t, t_dim> shape()
	{
		Array<size_t, t_dim> shp;
		fill_shape_(shp, 0);
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
		ArrayLike<Element_t, t_size_1, t_sizes...>::fill_shape_(shape, offset + 1);
	}
};

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class ArrayConstView : public ArrayLike<Element_t, t_size_0, t_sizes...>
{
public:
	ArrayConstView(const Element_t* elements) :
		m_elements(elements)
	{}

	const Element_t& operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return m_elements[pos];
	}

	//const Element_t& operator[](size_t pos)
	//{
	//	assert(pos < t_size_0);
	//	return m_elements[pos];
	//}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayConstView<Element_t, t_size>(m_elements);
	}

	//ArrayConstView<Element_t, t_size> flatten()
	//{
	//	return ArrayConstView<Element_t, t_size>(m_elements);
	//}
public:
	const Element_t* m_elements;
};


template<typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
class ArrayConstView<Element_t, t_size_0, t_size_1, t_sizes...> : public ArrayLike<Element_t, t_size_0, t_size_1, t_sizes...>
{
public:
	ArrayConstView(const Element_t* elements) :
		m_elements(elements)
	{}

	ArrayConstView<Element_t, t_size_1, t_sizes...> operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return ArrayConstView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	}

	//ArrayConstView<Element_t, t_size_1, t_sizes...> operator[](size_t pos)
	//{
	//	assert(pos < t_size_0);
	//	return ArrayConstView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	//}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayConstView<Element_t, t_size>(m_elements);
	}

	//ArrayConstView<Element_t, t_size> flatten()
	//{
	//	return ArrayConstView<Element_t, t_size>(m_elements);
	//}
public:
	const Element_t* m_elements;
};


template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class ArrayView : public ArrayLike<Element_t, t_size_0, t_sizes...>
{
public:
	ArrayView(Element_t* elements) :
		m_elements(elements)
	{}

	Element_t& operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return m_elements[pos];
	}

	const Element_t& operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return m_elements[pos];
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayConstView<Element_t, t_size>(m_elements);
	}
public:
	Element_t* m_elements;
};


template<typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
class ArrayView<Element_t, t_size_0, t_size_1, t_sizes...> : public ArrayLike<Element_t, t_size_0, t_size_1, t_sizes...>
{
public:
	ArrayView(Element_t* elements) :
		m_elements(elements)
	{}

	ArrayView<Element_t, t_size_1, t_sizes...> operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return ArrayView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	}

	ArrayConstView<Element_t, t_size_1, t_sizes...> operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return ArrayConstView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayConstView<Element_t, t_size>(m_elements);
	}
public:
	Element_t* m_elements;
};

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class Array
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	static const size_t t_dim = 1;
	static const size_t t_size = t_size_0;

	static Array<size_t, t_dim> shape()
	{
		Array<size_t, t_dim> shp;
		shp[0] = t_size_0;
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
	}
public:
	typedef ArrayView<Element_t, t_size_0, t_sizes...> View_t;
	typedef ArrayConstView<Element_t, t_size_0, t_sizes...> ConstView_t;
public:
	Element_t& operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return m_elements[pos];
	}

	const Element_t& operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return m_elements[pos];
	}

	ArrayView<Element_t, t_size_0, t_sizes...> view()
	{
		return ArrayView<Element_t, t_size_0, t_sizes...>(m_elements);
	}

	ArrayConstView<Element_t, t_size_0, t_sizes...> view() const
	{
		return ArrayConstView<Element_t, t_size_0, t_sizes...>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	bool operator==(const Array& other) const
	{
		return std::equal(&m_elements[0], &m_elements[t_size], &other.m_elements[0]);
	}

	bool operator!=(const Array& other) const
	{
		return !operator=(other);
	}
public:
	Element_t m_elements[t_size];
};

template<typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
class Array<Element_t, t_size_0, t_size_1, t_sizes...>
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	static const size_t t_sub_size = Array<Element_t, t_size_1, t_sizes...>::t_size;
	static const size_t t_sub_dims = Array<Element_t, t_size_1, t_sizes...>::t_dim;
	static const size_t t_dim = 1 + t_sub_dims;
	static const size_t t_size = t_size_0 * t_sub_size;

	static Array<size_t, t_dim> shape()
	{
		Array<size_t, t_dim> shp;
		fill_shape_(shp, 0);
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
		Array<Element_t, t_size_1, t_sizes...>::fill_shape_(shape, offset + 1);
	}
public:
	typedef ArrayView<Element_t, t_size_0, t_size_1, t_sizes...> View_t;
	typedef ArrayConstView<Element_t, t_size_0, t_size_1, t_sizes...> ConstView_t;
public:
	ArrayView<Element_t, t_size_1, t_sizes...> operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return ArrayView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	}

	ArrayConstView<Element_t, t_size_1, t_sizes...> operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return ArrayConstView<Element_t, t_size_1, t_sizes...>(&m_elements[pos * t_sub_size]);
	}

	ArrayView<Element_t, t_size_0, t_size_1, t_sizes...> view()
	{
		return ArrayView<Element_t, t_size_0, t_size_1, t_sizes...>(m_elements);
	}

	ArrayConstView<Element_t, t_size_0, t_size_1, t_sizes...> view() const
	{
		return ArrayConstView<Element_t, t_size_0, t_size_1, t_sizes...>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayConstView<Element_t, t_size> flatten() const
	{
		return ArrayConstView<Element_t, t_size>(m_elements);
	}

	bool operator==(const Array& other) const
	{
		return std::equal(&m_elements[0], &m_elements[t_size], &other.m_elements[0]);
	}

	bool operator!=(const Array& other) const
	{
		return !operator=(other);
	}

public:
	Element_t m_elements[t_size];
};


template<typename T>
struct Array_ElementType
{
	typedef T Element_t;
};

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
struct Array_ElementType<Array<Element_t, t_size_0, t_sizes...>>
{
	typedef Element_t Element_t;
};

template<typename T>
struct Array_Dimension
{
	//static const size_t t_dim = std::is_arithmetic_v<T> ? 1 : T::t_dim;
	constexpr static size_t dim()
	{
		if constexpr (std::is_arithmetic_v<T>)
		{
			return 1;
		}
		else
		{
			return T::t_dim;
		}
	}
};

template<typename T>
struct Array_Shape
{
	static Array<size_t, Array_Dimension<T>::dim()> shape()
	{
		if constexpr (std::is_arithmetic_v<T>)
		{
			Array<size_t, 1> shp;
			shp[0] = 1;
			return shp;
		}
		else
		{
			return T::shape();
		}
	}
};

template<typename Element_t, size_t t_size_0>
inline size_t Vector_dimension(const Array<Element_t, t_size_0>& arg)
{
	return t_size_0;
}

END_RLTL_IMPL
