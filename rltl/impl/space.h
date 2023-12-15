#pragma once
#include "utility.h"
#include "array.h"

BEGIN_RLTL_IMPL

struct DiscreteSpaceTag {};
struct DiscreteBoxSpaceTag{};
struct BoxSpaceTag {};
struct ImageSpaceTag {};

template<typename Integer_t = uint32_t>
class DiscreteSpace
{
public:
	typedef DiscreteSpaceTag SpaceTag_t;
	typedef Integer_t Numeric_t;
	typedef Integer_t Element_t;
public:
	DiscreteSpace(Element_t count):
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
	
	uint32_t dimension() const
	{
		return 1;
	}
	
	uint32_t count() const
	{
		return m_count;
	}
	
	bool valid(Element_t element) const
	{
		return element < m_count;
	}
protected:
	Element_t m_count;
};

template<typename Element_t>
class DiscreteBoxSpace
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