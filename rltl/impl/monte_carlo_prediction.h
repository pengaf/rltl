#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t, typename StateValueFunction_t, typename Policy_t>
class MonteCarloPrediction
{
public:
	MonteCarloPrediction(StateValueFunction_t& valueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f) :
		m_policy(policy),
		m_valueFunction(valueFunction),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
public:
	Action_t firstStep(State_t& firstState)
	{
		m_state = firstState;
		Action_t action = m_policy.takeAction(firstState);
		return action;
	}
	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_trajectory.emplace_back(m_state, reward);
		m_state = nextState;
		Action_t action = m_policy.takeAction(nextState);
		return action;
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		m_trajectory.emplace_back(m_state, reward);
		float g = 0;
		size_t count = m_trajectory.size();
		for (size_t i = 0; i < count; ++i)
		{
			SR& sr = m_trajectory[count - 1 - i];
			g = g * m_discountRate + sar.reward;
			float value = m_valueFunction.getValue(sar.state);
			float newValue = value + (g - value) * m_learningRate;
			m_valueFunction.setValue(sar.state, newValue);
		}
	}
protected:
	Policy_t& m_policy;
	ActionValueFunction_t& m_valueFunction;
	float m_learningRate;
	float m_discountRate;
protected:
	struct SR
	{
		State_t state;
		float reward;
		SR();
		SR(const State_t& s, const float& r) :
			state(s), reward(r)
		{}
	};
	std::vector<SAR> m_trajectory;
};

END_RLTL_IMPL