#pragma once
#include <vector>

namespace rltl
{
	class DiscreteSpace;

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

	template<typename V, typename S, typename A>
	class ActionValueFunction
	{
	public:
		typedef V value_type;
		typedef S state_type;
		typedef A action_type;
	public:
		virtual value_type getValue(const state_type& state, const action_type& action);
	};

	class StateValueTable : public StateValueFunction
	{
	public:
		typedef float value_type;
	public:
		StateValueTable(DiscreteSpace* space);
	public:
		value_type getValue(int32_t state) const
		{
			uint32_t index = state + m_start;
			assert(index < m_count);
			return m_values[index];
		}
		void setValue(int32_t state, value_type value)
		{
			uint32_t index = state + m_start;
			assert(index < m_count);
			m_values[index] = value;
		}
	protected:
		std::vector<value_type> m_values;
		int32_t m_start;
		uint32_t m_count;
	};

	class ActionValueTable
	{

	};
}
