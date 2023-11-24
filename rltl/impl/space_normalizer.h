#pragma once
#include "utility.h"
#include "space.h"
#include <array>

BEGIN_RLTL_IMPL

template<typename SrcInteger_t = int32_t, typename DstInteger_t = uint32_t>
class DiscreteToNormalizedDiscrete
{
public:
	typedef DiscreteSpace<SrcInteger_t> SrcSpace_t;
	typedef NormalizedDiscreteSpace<DstInteger_t> DstSpace_t;
	typedef typename SrcSpace_t::Element_t SrcElement_t;
	typedef typename DstSpace_t::Element_t DstElement_t;
public:
	DiscreteToNormalizedDiscrete(const SrcSpace_t& srcSpace) :
		m_srcSpace(srcSpace),
		m_dstSpace(DstInteger_t(srcSpace.count()))
	{}
public:
	const DstSpace_t& dstSpace() const
	{
		return m_dstSpace;
	}
	DstElement_t normalize(const SrcElement_t& src) const
	{
		assert(m_srcSpace.valid(src));
		DstElement_t dst = DstElement_t(src - m_srcSpace.m_begin);
		return dst;
	}
	SrcElement_t restore(const DstElement_t& dst) const
	{
		assert(m_dstSpace.valid(dst));
		SrcElement_t src = SrcElement_t(dst) + m_srcSpace.m_begin;
		return src;
	}
protected:
	const SrcSpace_t& m_srcSpace;
	DstSpace_t m_dstSpace;
};


template<size_t t_src_dimension, typename SrcInteger_t = int32_t, typename DstInteger_t = uint32_t>
class MultiDiscreteToNormalizedDiscrete
{
	static_assert(0 < t_src_dimension);
public:
	typedef MultiDiscreteSpace<t_src_dimension, SrcInteger_t> SrcSpace_t;
	typedef NormalizedDiscreteSpace<DstInteger_t> DstSpace_t;
	typedef typename SrcSpace_t::Element_t SrcElement_t;
	typedef typename DstSpace_t::Element_t DstElement_t;
public:
	MultiDiscreteToNormalizedDiscrete(const SrcSpace_t& srcSpace) :
		m_srcSpace(srcSpace),
		m_dstSpace(DstInteger_t(srcSpace.count()))
	{
		for (size_t i = 0; i < t_src_dimension; ++i)
		{
			m_counts[i] = DstInteger_t(srcSpace.count(i));
		}
	}
public:
	const DstSpace_t& dstSpace() const
	{
		return m_dstSpace;
	}
	DstElement_t normalize(const SrcElement_t& src) const
	{
		assert(m_srcSpace.valid(src));
		DstElement_t dst = DstElement_t(src[0] - m_srcSpace.begin(0));
		for (size_t i = 1; i < t_src_dimension; ++i)
		{
			dst = dst * m_counts[i] + DstElement_t(src[i] - m_srcSpace.begin(i));
		}
		return dst;
	}
	SrcElement_t restore(const DstElement_t& dst) const
	{
		assert(m_dstSpace.valid(dst));
		SrcElement_t src;
		DstElement_t remainder = dst;
		for (size_t i = t_src_dimension - 1; i > 0 ; --i)
		{
			src[i] = remainder % m_counts[i];
			remainder /= m_counts[i];
		}
		src[0] = remainder;
		return src;
	}
protected:
	const SrcSpace_t& m_srcSpace;
	DstSpace_t m_dstSpace;
	std::array<DstInteger_t, t_src_dimension> m_counts;
};


template<typename SrcInteger_t = uint32_t>
class NormalizedDiscreteToNormalizedDiscrete
{
public:
	typedef NormalizedDiscreteSpace<SrcInteger_t> SrcSpace_t;
	typedef NormalizedDiscreteSpace<SrcInteger_t> DstSpace_t;
	typedef typename SrcSpace_t::Element_t SrcElement_t;
	typedef typename DstSpace_t::Element_t DstElement_t;
public:
	NormalizedDiscreteToNormalizedDiscrete(const SrcSpace_t& space) :
		m_space(space)
	{}
public:
	const DstSpace_t& dstSpace() const
	{
		return m_srcSpace;
	}
	DstElement_t normalize(const SrcElement_t& src) const
	{
		assert(m_srcSpace.valid(src));
		return src;
	}
	SrcElement_t restore(const DstElement_t& dst) const
	{
		assert(m_srcSpace.valid(dst));
		return dst;
	}
protected:
	const SrcSpace_t& m_srcSpace;
};

template<size_t t_src_dimension, typename SrcReal_t = float, typename DstInteger_t = uint32_t>
class BoxToNormalizedDiscrete
{
public:
	typedef BoxSpace<t_src_dimension, SrcReal_t> SrcSpace_t;
	typedef NormalizedDiscreteSpace<DstInteger_t> DstSpace_t;
	typedef typename SrcSpace_t::Element_t SrcElement_t;
	typedef typename DstSpace_t::Element_t DstElement_t;
private:
	static DstInteger_t TotalDiscreteCount(const std::array<size_t, t_src_dimension>& discreteCount)
	{
		size_t totalDiscreteCount = 1;
		for (size_t i = 0; i < t_src_dimension; ++i)
		{
			assert(0 < discreteCounts[i]);
			totalDiscreteCount *= discreteCount[i];
		}
		return DstInteger_t(totalDiscreteCount);
	}
	static DstInteger_t TotalDiscreteCount(size_t discreteCountPerDimension)
	{
		size_t totalDiscreteCount = 1;
		for (size_t i = 0; i < t_src_dimension; ++i)
		{
			totalDiscreteCount *= discreteCountPerDimension;
		}
		return DstInteger_t(totalDiscreteCount);
	}
public:
	BoxToNormalizedDiscrete(const SrcSpace_t& srcSpace, const std::array<size_t, t_src_dimension>& discreteCount) :
		m_srcSpace(srcSpace),
		m_dstSpace(TotalDiscreteCount(discreteCount))
	{
		for (size_t i = 0; i < t_src_dimension; ++i)
		{
			m_discreteCount[i] = discreteCount[i];
		}
	}
	BoxToNormalizedDiscrete(const SrcSpace_t& srcSpace, size_t discreteCountPerDimension) :
		m_srcSpace(srcSpace),
		m_dstSpace(TotalDiscreteCount(discreteCountPerDimension))
	{
		assert(0 < discreteCountPerDimension);
		for (size_t i = 0; i < t_src_dimension; ++i)
		{
			m_discreteCount[i] = discreteCountPerDimension;
		}
	}
public:
	const DstSpace_t& dstSpace() const
	{
		return m_dstSpace;
	}
	DstElement_t normalize(const SrcElement_t& src) const
	{
		assert(m_srcSpace.valid(src));
		DstInteger_t dst = std::min<DstInteger_t>((src[0] - m_srcSpace.low(0)) / (m_srcSpace.high(0) - m_srcSpace.low(0)) * m_discreteCount[0], m_discreteCount[0]-1);
		assert(0 <= dst && dst < m_discreteCount[0]);
		for (size_t i = 1; i < t_src_dimension; ++i)
		{
			DstInteger_t offset = std::min<DstInteger_t>((src[i] - m_srcSpace.low(i)) / (m_srcSpace.high(i) - m_srcSpace.low(i)) * m_discreteCount[i], m_discreteCount[i] - 1);
			assert(0 <= offset && offset < m_discreteCount[0]);
			dst = dst * m_discreteCount[i] + offset;
		}
		return dst;
	}
	SrcElement_t restore(const DstElement_t& dst) const
	{
		assert(m_dstSpace.valid(dst));
		SrcElement_t src;
		DstElement_t remainder = dst;
		for (size_t i = t_src_dimension - 1; i > 0; --i)
		{
			src[i] = (remainder % m_discreteCount[i]) * (m_srcSpace.high(i) - m_srcSpace.low(i)) / m_discreteCount[i] + m_srcSpace.low(i);
			remainder /= m_discreteCount[i];
		}
		src[0] = remainder * (m_srcSpace.high(0) - m_srcSpace.low(0)) / m_discreteCount[i] + m_srcSpace.low(0);
		return src;
	}
protected:
	const SrcSpace_t & m_srcSpace;
	DstSpace_t m_dstSpace;
	std::array<DstInteger_t, t_src_dimension> m_discreteCount;
};

template<typename SpaceTag_t, typename SrcSpace_t, typename DstInteger_t = uint32_t>
struct ToNormalizedDiscrete_
{};

template<typename SrcSpace_t, typename DstInteger_t>
struct ToNormalizedDiscrete_<DiscreteSpaceTag, SrcSpace_t, DstInteger_t>
{
	typedef typename DiscreteToNormalizedDiscrete<typename SrcSpace_t::Element_t, DstInteger_t> Normalizer_t;
};

template<typename SrcSpace_t, typename DstInteger_t>
struct ToNormalizedDiscrete_<MultiDiscreteSpaceTag, SrcSpace_t, DstInteger_t>
{
	typedef typename MultiDiscreteToNormalizedDiscrete<SrcSpace_t::t_dimension, typename SrcSpace_t::Numeric_t, DstInteger_t> Normalizer_t;
};

template<typename SrcSpace_t, typename DstInteger_t>
struct ToNormalizedDiscrete_<NormalizedDiscreteSpaceTag, SrcSpace_t, DstInteger_t>
{
	typedef typename NormalizedDiscreteToNormalizedDiscrete<typename SrcSpace_t::Numeric_t> Normalizer_t;
};

template<typename SrcSpace_t, typename DstInteger_t>
struct ToNormalizedDiscrete_<BoxSpaceTag, SrcSpace_t, DstInteger_t>
{
	typedef typename BoxToNormalizedDiscrete<SrcSpace_t::t_dimension, typename SrcSpace_t::Numeric_t, DstInteger_t> Normalizer_t;
};

template<typename SrcSpace_t, typename DstInteger_t = uint32_t>
struct ToNormalizedDiscrete
{
	typedef typename ToNormalizedDiscrete_<typename SrcSpace_t::SpaceTag_t, SrcSpace_t, DstInteger_t>::Normalizer_t Normalizer_t;
};


END_RLTL_IMPL