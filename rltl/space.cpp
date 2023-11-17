#include "space.h"

namespace rltl
{
	DiscreteSpace::DiscreteSpace(const char* name, uint64_t count, int64_t start = 0) :
		Space(name),
		m_start(start),
		m_end(start + count)
	{
		if (INT8_MIN <= m_start && m_end <= INT8_MAX)
		{
			m_dataType = DataType::dt_int8;
		}
		else if (INT16_MIN <= m_start && m_end <= INT16_MAX)
		{
			m_dataType = DataType::dt_int16;
		}
		else if (INT32_MIN <= m_start && m_end <= INT32_MAX)
		{
			m_dataType = DataType::dt_int32;
		}
		else
		{
			m_dataType = DataType::dt_int64;
		}
	}

	ContinuousSpace::ContinuousSpace(const char* name, float low, float high, DataType dataType = DataType::dt_float32) :
		Space(name),
		m_low(low),
		m_high(high),
		m_dataType(dataType)
	{
		if(dataType != DataType::dt_float32)
		{
			m_dataType = DataType::dt_float64;
		}
	}
}