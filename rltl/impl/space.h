#pragma once
#include "utility.h"
#include "array.h"

BEGIN_RLTL_IMPL

struct DiscreteSpaceTag {};
struct MultiDiscreteSpaceTag {};
struct NormalizedDiscreteSpaceTag {};
struct BoxSpaceTag {};

enum class SpaceCategory
{
	discrete_space,
	multi_discrete_space,
	normalized_discrete_space,
	box_space,
};

template<typename Integer_t = int32_t>
class DiscreteSpace
{
public:
	typedef DiscreteSpaceTag SpaceTag_t;
	typedef Integer_t Numeric_t;
	typedef Integer_t Element_t;
public:
	DiscreteSpace(Element_t begin, Element_t end):
		m_begin(begin),
		m_end(end)
	{
		assert(begin < end);
	}
public:
	Integer_t begin() const
	{
		return m_begin;
	}
	Integer_t end() const
	{
		return m_end;
	}
	size_t dimension() const
	{
		return 1;
	}
	size_t count() const
	{
		return size_t(m_end - m_begin);
	}
	bool valid(Element_t element) const
	{
		return (m_begin <= element && element < m_end);
	}
public:
	Element_t m_begin;
	Element_t m_end;
};


template<size_t t_dimension, typename Integer_t = int32_t>
class MultiDiscreteSpace
{
public:
	typedef MultiDiscreteSpaceTag SpaceTag_t;
	typedef Integer_t Numeric_t;
	enum { t_dimension = t_dimension};
	typedef Array<Integer_t, t_dimension> Element_t;
public:
	MultiDiscreteSpace(Element_t begin, Element_t end) :
		m_begin(begin),
		m_end(end)
	{
		for (size_t i = 0; i < t_dimension; ++i)
		{
			assert(begin[i] < end[i]);
		}
	}
public:
	size_t dimension() const
	{
		return t_dimension;
	}
	const Element_t& begin() const
	{
		return m_begin;
	}
	const Element_t& end() const
	{
		return m_end;
	}
	Integer_t begin(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_begin[dim];
	}
	Integer_t end(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_end[dim];
	}
	size_t count(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_end[dim] - m_begin[dim];
	}
	size_t count() const
	{
		size_t n = 1;
		for (size_t i = 0; i < t_dimension; ++i)
		{
			n *= size_t(m_end[i] - m_begin[i]);
		}
		return n;
	}
	bool valid(const Element_t& element) const
	{
		for (size_t i = 0; i < t_dimension; ++i)
		{
			if (!(m_begin[i] <= element[i] && element[i] < m_end[i]))
			{
				return false;
			}
		}
		return true;
	}
public:
	Element_t m_begin;
	Element_t m_end;
};


template<typename Integer_t = uint32_t>
class NormalizedDiscreteSpace
{
public:
	typedef NormalizedDiscreteSpaceTag SpaceTag_t;
	typedef Integer_t Numeric_t;
	typedef Integer_t Element_t;
public:
	NormalizedDiscreteSpace(Element_t count) :
		m_count(count)
	{}
public:
	Integer_t begin() const
	{
		return 0;
	}
	Integer_t end() const
	{
		return m_count;
	}
	size_t dimension() const
	{
		return 1;
	}
	size_t count() const
	{
		return m_count;
	}
	bool valid(Element_t element) const
	{
		return (0 <= element && element < m_count);
	}
public:
	Element_t m_count;
};

template<size_t t_dimension, typename Real_t = float>
class BoxSpace
{
public:
	typedef BoxSpaceTag SpaceTag_t;
	typedef Real_t Numeric_t;
	enum { t_dimension = t_dimension };
	typedef Array<Real_t, t_dimension> Element_t;
public:
	BoxSpace(Element_t low, Element_t high) :
		m_low(low),
		m_high(high)
	{
		for (size_t i = 0; i < t_dimension; ++i)
		{
			assert(low[i] < high[i]);
		}
	}
public:
	size_t dimension() const
	{
		return t_dimension;
	}
	const Element_t& low() const
	{
		return m_low;
	}
	const Element_t& high() const
	{
		return m_high;
	}
	Real_t low(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_low[dim];
	}
	Real_t high(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_high[dim];
	}
	Real_t length(size_t dim) const
	{
		assert(dim < t_dimension);
		return m_high[dim] - m_low[dim];
	}
	bool valid(const Element_t& element) const
	{
		for (size_t i = 0; i < t_dimension; ++i)
		{
			if (!(m_low[i] <= element[i] && element[i] <= m_high[i]))
			{
				return false;
			}
		}
		return true;
	}
public:
	Element_t m_low;
	Element_t m_high;
};


END_RLTL_IMPL