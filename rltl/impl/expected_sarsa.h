#pragma once
#include "utility.h"
#include "epsilon_greedy.h"

BEGIN_RLTL_IMPL

template<typename ActionValueFunction_t, typename Policy_t = EpsilonGreedy<ActionValueFunction_t>>
class ExpectedSarsa
{
public:
	typedef typename ActionValueFunction_t::State_t State_t;
	typedef typename ActionValueFunction_t::Action_t Action_t;
public:
	ExpectedSarsa(ActionValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f) :
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
	Action_t nextStep(float reward, const State_t& nextState)
	{
		float value = m_valueFunction.getValue(m_state, m_action);
		std::vector<float> nextValues;
		m_valueFunction.getValues(nextValues, nextState);
		float target = reward + m_policy.getExpectedValue(nextValues) * m_discountRate;
		float newValue = value + (target - value) * m_learningRate;
		m_valueFunction.setValue(m_state, m_action, newValue);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		float value = m_valueFunction.getValue(m_state, m_action);
		float target = reward;
		if (nonterminal)
		{
			target += m_policy.getExpectedValue(nextState) * m_discountRate;
		}
		float newValue = value + (target - value) * m_learningRate;
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
inline static ExpectedSarsa<ActionValueFunction_t, Policy_t> MakeExpectedSarsa(ActionValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f)
{
	return ExpectedSarsa(valueFunction, policy, learningRate, discountRate);
}

END_RLTL_IMPL