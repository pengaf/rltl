#pragma once
#include "utility.h"
#include <array>
#include <vector>
#include "array.h"

BEGIN_RLTL_IMPL

enum class ScalarType
{
	st_int32 = 0,
	st_float32 = 1,
};

class NumArray;
class NumArrayImpl
{
	friend class NumArray;
	struct
	{
		uint32_t m_scalarType : 1;
		uint32_t m_dimension : 31;
	};
public:
	ScalarType scalarType() const
	{
		return (ScalarType)m_scalarType;
	}

	uint32_t dim() const
	{
		return m_dimension;
	}

	const uint32_t* sizes() const
	{
		return (uint32_t*)(this + 1);
	}

	uint32_t size(uint32_t dim) const
	{
		const uint32_t* s = sizes();
		if (dim < m_dimension)
		{
			return s[dim];
		}
		return 1;
	}

	void* data()
	{
		return (uint32_t*)(this + 1) + m_dimension;
	}

public:
	static size_t ScalarSize(ScalarType scalarType)
	{
		switch (scalarType)
		{
		case ScalarType::st_int32:
			return sizeof(int32_t);
		case ScalarType::st_float32:
			return sizeof(float);
		}
		return 0;
	}
public:
	static NumArrayImpl* New(ScalarType scalarType, const uint32_t* sizes, uint32_t dim)
	{
		size_t scalarSize = ScalarSize(scalarType);
		if (0 == scalarSize)
		{
			return nullptr;
		}
		size_t numElements = 1;
		for (uint32_t i = 0; i < dim; ++i)
		{
			numElements *= sizes[i];
		}
		size_t bufferSize = sizeof(NumArrayImpl) + sizeof(uint32_t) * dim + scalarSize * numElements;

		void* buffer = malloc(bufferSize);
		NumArrayImpl* impl = (NumArrayImpl*)buffer;
		impl->m_scalarType = (uint32_t)scalarType;
		impl->m_dimension = dim;
		uint32_t* dstSizes = (uint32_t*)(impl + 1);
		for (uint32_t i = 0; i < dim; ++i)
		{
			dstSizes[i] = sizes[i];
		}
		return impl;
	}
};

class NumArray
{
public:
	NumArray(ScalarType scalarType, const uint32_t* sizes, uint32_t dim)
	{
		m_impl = NumArrayImpl::New(scalarType, sizes, dim);
	}

	NumArray(ScalarType scalarType, const std::vector<uint32_t>& sizes):
		NumArray(scalarType, sizes.data(), sizes.size())
	{}

	template<uint32_t t_dim>
	NumArray(ScalarType scalarType, const std::array<uint32_t, t_dim>& sizes) :
		NumArray(scalarType, sizes.data(), t_dim)
	{}

	NumArray(const NumArray& other) = delete;

	NumArray(NumArray&& other) :
		m_impl(other.m_impl)
	{
		other.m_impl = nullptr;
	}

	~NumArray()
	{
		free(m_impl);
	}

	NumArray& operator=(NumArray&& other)
	{
		free(m_impl);
		m_impl = other.m_impl;
		other.m_impl = nullptr;
		return *this;
	}

	NumArray& operator=(const NumArray& other) = delete;
public:
	ScalarType scalarType() const
	{
		return m_impl->scalarType();
	}

	uint32_t dim() const
	{
		return m_impl->dim();
	}

	uint32_t size(uint32_t dim) const
	{
		return m_impl->size(dim);
	}

	void* data()
	{
		return m_impl->data();
	}

private:
	NumArrayImpl* m_impl;
};

END_RLTL_IMPL
