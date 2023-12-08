#pragma once
#include "utility.h"
#include <array>
#include <vector>
#include "array.h"


BEGIN_RLTL_IMPL

enum class ScalarType
{
	st_int8,
	st_uint8,
	st_int16,
	st_uint16,
	st_int32,
	st_uint32,
	st_int64,
	st_uint64,
	st_float,
	st_double,
	st_count,
};

//struct DynarrayMeta
//{
//	ScalarType m_elementType;
//	uint32_t m_dim;
//	uint32_t m_
//};
//

class Dynarray
{
public:
	template<uint32_t t_dim>
	Dynarray(ScalarType scalarType, const std::array<uint32_t, t_dim>& sizes):
		Dynarray(scalarType, sizes.data(), t_dim)
	{}

	Dynarray(ScalarType scalarType, const std::vector<uint32_t>& sizes):
		Dynarray(scalarType, sizes.data(), sizes.size())
	{}

	Dynarray(ScalarType scalarType, const uint32_t* sizes, uint32_t dim):
		m_scalarType(scalarType),
		m_dim(dim)
	{
		size_t numElements = 1;
		for (uint32_t i = 0; i < dim; ++i)
		{
			numElements *= sizes[i];
		}
		size_t bufferSize = align8(sizeof(uint32_t) * dim) + scalarSize(scalarType) * numElements;
		if (bufferSize)
		{
			m_buffer = malloc(bufferSize);
			for (uint32_t i = 0; i < dim; ++i)
			{
				((uint32_t*)m_buffer)[i] = sizes[i];
			}
		}
	}
	
	Dynarray(const Dynarray& other) = delete;
	
	Dynarray(Dynarray&& other):
		m_scalarType(other.m_scalarType),
		m_dim(other.m_dim),
		m_buffer(other.m_buffer)
	{
		other.m_buffer = nullptr;
		other.m_dim = 0;
		other.m_scalarType = ScalarType::st_count;
	}

	~Dynarray()
	{
		free(m_buffer);
	}

	Dynarray& operator=(Dynarray&& other)
	{
		free(m_buffer);
		m_scalarType = other.m_scalarType;
		m_dim = other.m_dim;
		m_buffer = other.m_buffer;
		other.m_buffer = nullptr;
		other.m_dim = 0;
		other.m_scalarType = ScalarType::st_count;
		return *this;
	}

	Dynarray& operator=(const Dynarray& other) = delete;
public:
	ScalarType scalarType() const
	{
		return m_scalarType;
	}

	uint32_t dim() const
	{
		return m_dim;
	}

	uint32_t size(uint32_t dim) const
	{
		if (dim < m_dim)
		{
			return ((uint32_t*)m_buffer)[dim];
		}
		return 0;
	}

	void* data()
	{
		return (uint32_t*)m_buffer + m_dim;
	}
private:
	static size_t align8(size_t n)
	{
		return (n + 7) / 8 * 8;
	}
	static size_t scalarSize(ScalarType scalarType)
	{
		switch (scalarType)
		{
		case ScalarType::st_int8:
		case ScalarType::st_uint8:
			return 1;
		case ScalarType::st_int16:
		case ScalarType::st_uint16:
			return 2;
		case ScalarType::st_int32:
		case ScalarType::st_uint32:
			return 4;
		case ScalarType::st_int64:
		case ScalarType::st_uint64:
			return 8;
		case ScalarType::st_float:
			return sizeof(float);
		case ScalarType::st_double:
			return sizeof(float);
		}
		return 0;
	}
private:
	ScalarType m_scalarType;
	uint32_t m_dim;
	void* m_buffer{};
};

END_RLTL_IMPL
