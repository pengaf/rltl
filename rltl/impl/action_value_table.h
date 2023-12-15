#pragma once
#include "utility.h"
#include <vector>

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class ActionValueTable : public ActionValueFunction<State_t, Action_t>
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;	
public:
	ActionValueTable(uint32_t stateCount, uint32_t actionCount) :
		m_stateCount(stateCount),
		m_actionCount(actionCount),
		m_values(stateCount * actionCount)
	{}	
public:
	//for policy (greedy, epsilon greedy, etc)
	Action_t firstMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
		uint32_t index = state * m_actionCount;
		float maxValue = m_values[index];
		uint32_t maxIndex = 0;
		for (uint32_t i = 1; i < m_actionCount; ++i)
		{
			float value = m_values[index + i];
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
	//expected sarsa
	void getValues(std::vector<float>& values, const State_t& state) const
	{
		values.resize(m_actionCount);
		for (uint32_t i = 0; i < m_actionCount; ++i)
		{
			values[i] = m_values[state * m_actionCount + i];
		}
	}
public:
	//for tabular agent
	float getValue(const State_t& state, const Action_t& action) const
	{
		assert(state < m_stateCount && action < m_actionCount);
		return m_values[state * m_actionCount + action];
	}
	void setValue(const State_t& state, const Action_t& action, const float& value)
	{
		assert(state < m_stateCount && action < m_actionCount);
		m_values[state * m_actionCount + action] = value;
	}
	float getMaxValue(const State_t& state) const
	{
		assert(state < m_stateCount);
		uint32_t index = state * m_actionCount;
		float maxValue = m_values[index];
		for (uint32_t i = 1; i < m_actionCount; ++i)
		{
			float value = m_values[index + i];
			if (maxValue < value)
			{
				maxValue = value;
			}
		}
		return maxValue;
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
	std::vector<float> m_values;
};

END_RLTL_IMPL
