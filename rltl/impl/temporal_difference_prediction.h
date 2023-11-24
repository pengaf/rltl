#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename Policy_t, typename StateValueFunction_t>
class TemporalDifferencePrediction
{
public:
	typedef typename StateValueFunction_t::State_t State_t;
	typedef typename StateValueFunction_t::Value_t Reward_t;
	typedef typename Policy_t::Action_t Action_t;
public:
	TemporalDifferencePrediction(StateValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f) :
		m_valueFunction(valueFunction),
		m_policy(policy),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
public:
	Action_t firstStep(State_t& firstState)
	{
		m_state = firstState;
		Action_t action = m_policy.takeAction(firstState);
		return action;
	}
	Action_t nextStep(const Reward_t& reward, const State_t& nextState)
	{
		ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state);
		ActionValueFunction_t::Value_t target = reward + m_valueFunction.getValue(nextState) * m_discountRate;
		ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
		m_valueFunction.setValue(m_state, newValue);
		m_state = nextState;
		Action_t action = m_policy.takeAction(nextState);
		return action;
	}
	void lastStep(const Reward_t& reward, const State_t& nextState, bool terminated)
	{
		ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state);
		ActionValueFunction_t::Value_t target = reward;
		if (!terminated)
		{
			target += m_valueFunction.getValue(nextState) * m_discountRate;
		}
		ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
		m_valueFunction.setValue(m_state, newValue);
	}
protected:
	Policy_t& m_policy;
	ActionValueFunction_t& m_valueFunction;
	float m_learningRate;
	float m_discountRate;
};

END_RLTL_IMPL