#pragma once
#include "utility.h"
#include <array>

BEGIN_RLTL_IMPL

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class Array;

template<typename Element_t, size_t t_size_0, size_t... t_sizes>
class ArrayView
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	static const size_t t_dims = 1;
	static const size_t t_size = t_size_0;
public:
	ArrayView(Element_t* elements) :
		m_elements(elements)
	{}
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

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten() const
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	static Array<size_t, t_dims> shape()
	{
		Array<size_t, t_dims> shp;
		shp[0] = t_size_0;
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
	}

public:
	Element_t* m_elements;
};

template<typename Element_t, size_t t_size_0, size_t t_size_1, size_t... t_sizes>
class ArrayView<Element_t, t_size_0, t_size_1, t_sizes...>
{
public:
	typedef Element_t Element_t;
	static const size_t t_size_0 = t_size_0;
	typedef ArrayView<Element_t, t_size_1, t_sizes...> SubView_t;
	static const size_t t_sub_size = SubView_t::t_size;
	static const size_t t_sub_dims = SubView_t::t_dims;
	static const size_t t_dims = 1 + t_sub_dims;
	static const size_t t_size = t_size_0 * t_sub_size;
public:
	ArrayView(Element_t* elements) :
		m_elements(elements)
	{}
public:
	SubView_t operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return SubView_t(&m_elements[pos * t_sub_size]);
	}

	SubView_t operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return SubView_t(&m_elements[pos * t_sub_size]);
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten() const
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	static Array<size_t, t_dims> shape()
	{
		Array<size_t, t_dims> shp;
		fill_shape_(shp, 0);
		return shp;
	}

	template<size_t n>
	static void fill_shape_(Array<size_t, n>& shape, size_t offset)
	{
		shape[offset] = t_size_0;
		SubView_t::fill_shape_(shape, offset + 1);
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
	static const size_t t_dims = 1;
	static const size_t t_size = t_size_0;
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

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten() const
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	static Array<size_t, t_dims> shape()
	{
		return ArrayView<Element_t, t_size, t_sizes...>::shape();
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
	typedef ArrayView<Element_t, t_size_1, t_sizes...> SubView_t;
	static const size_t t_sub_size = SubView_t::t_size;
	static const size_t t_sub_dims = SubView_t::t_dims;
	static const size_t t_dims = 1 + t_sub_dims;
	static const size_t t_size = t_size_0 * t_sub_size;
public:
	SubView_t operator[](size_t pos)
	{
		assert(pos < t_size_0);
		return SubView_t(&m_elements[pos * t_sub_size]);
	}

	SubView_t operator[](size_t pos) const
	{
		assert(pos < t_size_0);
		return SubView_t(&m_elements[pos * t_sub_size]);
	}

	ArrayView<Element_t, t_size> flatten()
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	ArrayView<Element_t, t_size> flatten() const
	{
		return ArrayView<Element_t, t_size>(m_elements);
	}

	static Array<size_t, t_dims> shape()
	{
		return ArrayView<Element_t, t_size_0, t_size_1, t_sizes...>::shape();
	}
public:
	Element_t m_elements[t_size];
};

END_RLTL_IMPL
