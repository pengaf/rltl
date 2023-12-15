#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class Sarsa
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef ActionValueFunction<State_t, Action_t> ActionValueFunction_t;
	typedef PolicyFunction<State_t, Action_t> PolicyFunction_t;
	typedef paf::SharedPtr<ActionValueFunction_t> ActionValueFunctionPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef paf::SharedPtr<Sarsa> SarsaPtr;
public:
	Sarsa(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f) :
		m_actionValueFunction(actionValueFunction),
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
		float value = m_actionValueFunction.getValue(m_state, m_action);
		Action_t nextAction = m_policy.takeAction(nextState);
		float target = reward + m_actionValueFunction.getValue(nextState, nextAction) * m_discountRate;
		float newValue = value + (target - value) * m_learningRate;
		m_actionValueFunction.setValue(m_state, m_action, newValue);
		m_state = nextState;
		m_action = nextAction;
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		float value = m_actionValueFunction.getValue(m_state, m_action);
		float target = reward;
		if (!terminated)
		{
			Action_t nextAction = m_policy.takeAction(nextState);
			target += m_actionValueFunction.getValue(nextState, nextAction) * m_discountRate;
		}
		float newValue = value + (target - value) * m_learningRate;
		m_actionValueFunction.setValue(m_state, m_action, newValue);
	}
protected:
	ActionValueFunctionPtr m_actionValueFunction;
	PolicyFunctionPtr m_policy;
	float m_learningRate;
	float m_discountRate;
protected:
	State_t m_state;
	Action_t m_action;
public:
	static SarsaPtr Make(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f)
	{
		return SarsaPtr::Make(actionValueFunction, policy, learningRate, discountRate);
	}
};

//template<typename ActionValueFunction_t, typename Policy_t = EpsilonGreedy<ActionValueFunction_t>>
//inline static Sarsa<ActionValueFunction_t, Policy_t> MakeSarsa(ActionValueFunction_t& actionValueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f)
//{
//	return Sarsa(actionValueFunction, policy, learningRate, discountRate);
//}

END_RLTL_IMPL