#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class MonteCarloControl
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef ActionValueFunction<State_t, Action_t> ActionValueFunction_t;
	typedef PolicyFunction<State_t, Action_t> PolicyFunction_t;
	typedef paf::SharedPtr<ActionValueFunction_t> ActionValueFunctionPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef paf::SharedPtr<MonteCarloControl> MonteCarloControlPtr;
public:
	MonteCarloControl(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f) :
		m_actionValueFunction(actionValueFunction),
		m_policy(policy),
		m_learningRate(learningRate),
		m_discountRate(discountRate)
public:
	Action_t firstStep(State_t& firstState)
	{
		m_state = firstState;
		m_action = m_policy.takeAction(firstState);
		return m_action;
	}
	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		m_trajectory.emplace_back(m_state, m_action, reward);
		float g = 0;
		size_t count = m_trajectory.size();
		for (size_t i = 0; i < count; ++i)
		{
			SAR& sar = m_trajectory[count - 1 - i];
			g = g * m_discountRate + sar.reward;
			float value = m_actionValueFunction.getValue(sar.state, sar.action);
			float newValue = value + (g - value) * m_learningRate;
			m_actionValueFunction.setValue(sar.state, sar.action, newValue);
		}
	}
protected:
	ActionValueFunctionPtr m_actionValueFunction;
	PolicyFunctionPtr m_policy;
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
public:
	static MonteCarloControlPtr Make(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f)
	{
		return MonteCarloControlPtr::Make(actionValueFunction, policy, learningRate, discountRate);
	}
};

END_RLTL_IMPL