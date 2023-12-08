#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t, typename ActionValueFunction_t, typename Explorator_t>
class MonteCarloControl
{
public:
	MonteCarloControl(ActionValueFunction_t& valueFunction, Explorator_t& explorator, float learningRate, float discountRate = 1.0f) :
		m_explorator(explorator),
		m_valueFunction(valueFunction),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
public:
	Action_t firstStep(State_t& firstState)
	{
		m_state = firstState;
		m_action = m_explorator.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		m_state = nextState;
		m_action = m_explorator.takeAction(nextState);
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool nonterminal)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		float g = 0;
		size_t count = m_trajectory.size();
		for (size_t i = 0; i < count; ++i)
		{
			SAR& sar = m_trajectory[count - 1 - i];
			g = g * m_discountRate + sar.reward;
			float value = m_valueFunction.getValue(sar.state, sar.action);
			float newValue = value + (g - value) * m_learningRate;
			m_valueFunction.setValue(sar.state, sar.action, newValue);
		}
	}
protected:
	Explorator_t& m_explorator;
	ActionValueFunction_t& m_valueFunction;
	float m_learningRate;
	float m_discountRate;
protected:
	struct SAR
	{
		State_t state;
		Action_t action;
		float reward;
		SAR();
		SAR(const State_t& s, const Acton_t& a, const float& r) :
			state(s), action(a), reward(r)
		{}
	};
	std::vector<SAR> m_trajectory;
};

END_RLTL_IMPL