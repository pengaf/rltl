#pragma once
#include "utility.h"
#include <vector>

BEGIN_RLTL_IMPL

template<typename State_t = uint32_t, typename Action_t = uint32_t, typename Value_t = float>
class ActionValueTable
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Value_t Value_t;
public:
	ActionValueTable(uint32_t stateCount, uint32_t actionCount) :
		m_stateCount(stateCount),
		m_actionCount(actionCount),
		m_values(stateCount * actionCount)
	{}
public:
	Value_t getValue(const State_t& state, const Action_t& action) const
	{
		assert(state < m_stateCount && action < m_actionCount);
		return m_values[state * m_actionCount + action];
	}
	void setValue(const State_t& state, const Action_t& action, const Value_t& value)
	{
		assert(state < m_stateCount && action < m_actionCount);
		m_values[state * m_actionCount + action] = value;
	}
	Value_t firstMaxValue(const State_t& state) const
	{
		assert(state < m_stateCount);
		uint32_t index = state * m_actionCount;
		Value_t maxValue = m_values[index];
		for (uint32_t i = 1; i < m_actionCount; ++i)
		{
			Value_t value = m_values[index + i];
			if (maxValue < value)
			{
				maxValue = value;
			}
		}
		return maxValue;
	}
	Value_t randomMaxValue(const State_t& state) const
	{
		//assert(state < m_stateCount);
		return firstMaxValue(state);
	}
	Value_t meanValue(const State_t& state) const
	{
		uint32_t index = state * m_actionCount;
		Value_t valueSum = m_values[index];
		for (uint32_t i = 1; i < m_actionCount; ++i)
		{
			Value_t value = m_values[index + i];
			valueSum += value;
		}
		return valueSum / float(m_actionCount);
	}
	Action_t firstMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
		uint32_t index = state * m_actionCount;
		Value_t maxValue = m_values[index];
		uint32_t maxIndex = 0;
		for (uint32_t i = 1; i < m_actionCount; ++i)
		{
			Value_t value = m_values[index + i];
			if (maxValue < value)
			{
				maxValue = value;
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	Action_t randomMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
		return firstMaxAction(state);
	}
	uint32_t stateCount() const
	{
		return m_stateCount;
	}
	uint32_t actionCount() const
	{
		return m_actionCount;
	}
protected:
	uint32_t m_stateCount;
	uint32_t m_actionCount;
	std::vector<Value_t> m_values;
};

END_RLTL_IMPL
