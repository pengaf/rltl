#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t, typename Reward_t, typename ActionValueFunction_t, typename Explorator_t>
class Reinforce
{
public:
	Reinforce(PolicyFunction_t& policyFunction, float learningRate, float discountRate = 1.0f) :
		m_policyFunction(policyFunction),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
public:
	Action_t firstStep(State_t& firstState)
	{
		m_state = firstState;
		m_action = m_explorator.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(Reward_t reward, const State_t& nextState)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		m_state = nextState;
		m_action = m_explorator.takeAction(nextState);
		return m_action;
	}
	void lastStep(Reward_t reward, const State_t& nextState, bool nonterminal)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		StateValueFunction_t::Value_t g = 0;
		size_t count = m_trajectory.size();
		for (size_t i = 0; i < count; ++i)
		{
			SAR& sar = m_trajectory[count - 1 - i];
			g = g * m_discountRate + sar.reward;
			sar.g = g;
		}
		float gamma_n = 1.0f;
		for (size_t i = 0; i < count; ++i)
		{
			SAR& sar = m_trajectory[i];
			m_policyFunction.update(m_learningRate * gamma_n * sar.g);
			gamma_n *= m_discountRate;
		}
	}
protected:
	PolicyFunction_t& m_policyFunction;
	float m_learningRate;
	float m_discountRate;
protected:
	struct SAR
	{
		State_t state;
		Action_t action;
		Reward_t reward;
		float g;
		SAR();
		SAR(const State_t& s, const Acton_t& a, const Reward_t& r) :
			state(s), action(a), reward(r)
		{}
	};
	std::vector<SAR> m_trajectory;
};


END_RLTL_IMPL
