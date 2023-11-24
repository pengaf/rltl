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



END_RLTL_IMPL