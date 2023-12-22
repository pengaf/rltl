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
	ScalarType m_scalarType;
	uint32_t m_dimension;
	size_t m_numElements;
public:
	ScalarType scalarType() const
	{
		return ScalarType(m_scalarType);
	}

	uint32_t dim() const
	{
		return m_dimension;
	}
	
	size_t numel() const
	{
		return m_numElements;
	}

	const size_t* sizes() const
	{
		return (size_t*)(this + 1);
	}

	size_t size(uint32_t dim) const
	{
		const size_t* s = sizes();
		if (dim < m_dimension)
		{
			return s[dim];
		}
		return 1;
	}

	void* data()
	{
		return (size_t*)(this + 1) + m_dimension;
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
private:
	size_t memorySize() const
	{
		size_t scalarSize = ScalarSize(scalarType());
		size_t bufferSize = sizeof(NumArrayImpl) + sizeof(size_t) * m_dimension + scalarSize * m_numElements;
		return bufferSize;
	}
public:
	static NumArrayImpl* New(ScalarType scalarType, const size_t* sizes, uint32_t dim)
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
		size_t bufferSize = sizeof(NumArrayImpl) + sizeof(size_t) * dim + scalarSize * numElements;

		void* buffer = malloc(bufferSize);
		NumArrayImpl* impl = (NumArrayImpl*)buffer;
		impl->m_scalarType = scalarType;
		impl->m_dimension = dim;
		impl->m_numElements = numElements;
		size_t* dstSizes = (size_t*)(impl + 1);
		for (uint32_t i = 0; i < dim; ++i)
		{
			dstSizes[i] = sizes[i];
		}
		return impl;
	}
};

template<typename Scalar_t, uint32_t t_dim>
class NumArrayView
{
public:
	NumArrayView(Scalar_t* data, size_t* sizes, size_t stride):
		m_data(data),
		m_sizes(sizes),
		m_stride(stride)
	{
		assert(stride % m_sizes[1] == 0);
	}
	NumArrayView<Scalar_t, t_dim - 1> operator[](size_t pos)
	{
		assert(pos < m_sizes[0]);
		return NumArrayView<Scalar_t, t_dim - 1>(m_data + pos * m_stride, m_sizes + 1, m_stride/m_sizes[1])
		//return m_data[pos];
	}
private:
	Scalar_t* m_data;
	size_t* m_sizes;
	size_t m_stride;
};

template<typename Scalar_t>
class NumArrayView<Scalar_t, 1>
{
public:
	NumArrayView(Scalar_t* data, size_t* sizes, size_t stride) :
		m_data(data),
		m_size(sizes[0])
	{
		assert(1 == stride);
	}
public:
	Scalar_t& operator[](size_t pos)
	{
		assert(pos < m_size);
		return m_data[pos];
	}

	const Scalar_t& operator[](size_t pos) const
	{
		assert(pos < m_size);
		return m_data[pos];
	}
private:
	Scalar_t* m_data;
	size_t m_size;
};


class NumArray
{
public:
	NumArray() :
		m_impl(nullptr)
	{}

	NumArray(ScalarType scalarType, const size_t* sizes, uint32_t dim)
	{
		m_impl = NumArrayImpl::New(scalarType, sizes, dim);
	}

	NumArray(ScalarType scalarType, const std::vector<size_t>& sizes):
		NumArray(scalarType, sizes.data(), sizes.size())
	{}

	template<size_t t_dim>
	NumArray(ScalarType scalarType, const std::array<size_t, t_dim>& sizes) :
		NumArray(scalarType, sizes.data(), t_dim)
	{}

	NumArray(const NumArray& other)
	{
		if (other.m_impl)
		{
			size_t size = other.m_impl->memorySize();
			void* buffer = malloc(size);
			memcpy(buffer, other.m_impl, size);
			NumArrayImpl* impl = (NumArrayImpl*)buffer;
		}
		else
		{
			m_impl = nullptr;
		}
	}

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

	NumArray& operator=(const NumArray& other)
	{
		if (this != &other)
		{
			free(m_impl);
			if (other.m_impl)
			{
				size_t size = other.m_impl->memorySize();
				void* buffer = malloc(size);
				memcpy(buffer, other.m_impl, size);
				NumArrayImpl* impl = (NumArrayImpl*)buffer;
			}
			else
			{
				m_impl = nullptr;
			}
		}
	}
	template<typename Scalar_t, uint32_t t_dim> 
	NumArrayView<Scalar_t, t_dim> view()
	{
		assert(m_impl && m_impl->dim() == t_dim);
		return NumArrayView<Scalar_t, t_dim>((Scalar_t*)m_impl->data(), m_impl->sizes(), m_impl->numel() / m_impl->sizes()[0]);
	}
public:
	ScalarType scalarType() const
	{
		return m_impl->scalarType();
	}

	uint32_t dim() const
	{
		return m_impl->dim();
	}

	size_t size(uint32_t dim) const
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
