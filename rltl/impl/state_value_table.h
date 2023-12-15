#pragma once
#include "utility.h"
#include <stdint.h>
#include <assert.h>
#include <vector>

BEGIN_RLTL_IMPL

template<typename State_t>
class StateValueTable
{
public:
	typedef State_t State_t;
public:
	StateValueTable(uint32_t stateCount) :
		m_values(stateCount)
	{}
public:
	float getValue(const State_t& state) const
	{
		assert(state < m_values.size());
		return m_values[state];
	}
	void setValue(const State_t& state, const float& value)
	{
		assert(state < m_values.size());
		m_values[state] = value;
	}
protected:
	std::vector<float> m_values;
};

END_RLTL_IMPL