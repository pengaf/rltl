#pragma once
#include "utility.h"
#include "epsilon_greedy.h"

BEGIN_RLTL_IMPL

template<typename ActionValueFunction_t, typename Policy_t = EpsilonGreedy<ActionValueFunction_t>>
class QLearning
{
public:
	typedef typename ActionValueFunction_t::State_t State_t;
	typedef typename ActionValueFunction_t::Value_t Reward_t;
	typedef typename ActionValueFunction_t::Action_t Action_t;
public:
	QLearning(ActionValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f) :
		m_valueFunction(valueFunction),
		m_policy(policy),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
	{}
public:
	Action_t firstStep(const State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(const Reward_t& reward, const State_t& nextState)
	{
		ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state, m_action);
		ActionValueFunction_t::Value_t target = reward + m_valueFunction.firstMaxValue(nextState) * m_discountRate;
		ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
		m_valueFunction.setValue(m_state, m_action, newValue);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(const Reward_t& reward, const State_t& nextState, bool terminated)
	{
		ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state, m_action);
		ActionValueFunction_t::Value_t target = reward;
		if (!terminated)
		{
			target += m_valueFunction.firstMaxValue(nextState) * m_discountRate;
		}
		ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
		m_valueFunction.setValue(m_state, m_action, newValue);
	}
protected:
	ActionValueFunction_t& m_valueFunction;
	Policy_t& m_policy;
	float m_learningRate;
	float m_discountRate;
protected:
	State_t m_state;
	Action_t m_action;
};

template<typename ActionValueFunction_t, typename Policy_t = EpsilonGreedy<ActionValueFunction_t>>
inline static QLearning<ActionValueFunction_t, Policy_t> MakeQLearning(ActionValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f)
{
	return QLearning(valueFunction, policy, learningRate, discountRate);
}

END_RLTL_IMPL