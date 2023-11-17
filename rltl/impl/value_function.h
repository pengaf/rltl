#pragma once
#include "utility.h"
#include <stdint.h>
#include <assert.h>
#include <vector>

BEGIN_RLTL_IMPL

/*

template<typename State_t, typename Value_t = float>
class StateValueFunction
{
public:
	typedef State_t State_t;
	typedef Value_t Value_t;
public:
	Value_t getValue(const State_t& state);
	void setValue(const State_t& state, const Value_t& value);
};

template<typename State_t, typename Action_t, typename Value_t = float>
class ActionValueFunction
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Value_t Value_t;
public:
	Value_t getValue(const State_t& state, const Action_t& action);
	void setValue(const State_t& state, const Action_t& action, const Value_t& value);
};

*/

template<typename State_t = uint32_t, typename Value_t = float>
class StateValueTable
{
public:
	typedef State_t State_t;
	typedef Value_t Value_t;
public:
	StateValueTable(uint32_t stateCount) :
		m_values(stateCount)
	{}
public:
	value_type getValue(const State_t& state) const
	{
		assert(state < m_values.size());
		return m_values[state];
	}
	void setValue(const State_t& state, const Value_t& value)
	{
		assert(state < m_values.size());
		m_values[state] = value;
	}
protected:
	std::vector<Value_t> m_values;
};

template<typename State_t = uint32_t, typename Action_t = uint32_t, typename Value_t = float>
class ActionValueTable
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef Value_t Value_t;
public:
	StateValueTable(uint32_t stateCount, uint32_t actionCount) :
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
	}
	Value_t randomMaxValue(const State_t& state) const
	{
		assert(state < m_stateCount);
	}
	Value_t meanValue(const State_t& state) const
	{
		assert(state < m_stateCount);
	}
	Action_t firstMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
	}
	Action_t randomMaxAction(const State_t& state) const
	{
		assert(state < m_stateCount);
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