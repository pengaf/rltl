#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class QLearning
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef ActionValueFunction<State_t, Action_t> ActionValueFunction_t;
	typedef PolicyFunction<State_t, Action_t> PolicyFunction_t;
	typedef paf::SharedPtr<ActionValueFunction_t> ActionValueFunctionPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef paf::SharedPtr<QLearning> QLearningPtr;
public:
	QLearning(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f) :
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
		float target = reward + m_actionValueFunction.getMaxValue(nextState) * m_discountRate;
		float newValue = value + (target - value) * m_learningRate;
		m_actionValueFunction.setValue(m_state, m_action, newValue);
		m_state = nextState;
		m_action = m_policy.takeAction(nextState);
		return m_action;
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		float value = m_actionValueFunction.getValue(m_state, m_action);
		float target = reward;
		if (!terminated)
		{
			target += m_actionValueFunction.getMaxValue(nextState) * m_discountRate;
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
	static QLearningPtr Make(ActionValueFunctionPtr actionValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f)
	{
		return QLearningPtr::Make(actionValueFunction, policy, learningRate, discountRate);
	}
};

//template<typename State_t, typename Action_t>
//inline static QLearning<State_t, Action_t>::QLearningPtr MakeQLearning(ActionValueFunction_t& actionValueFunction, Policy_t& policy, float learningRate, float discountRate = 1.0f)
//{
//	return QLearning(actionValueFunction, policy, learningRate, discountRate);
//}

END_RLTL_IMPL