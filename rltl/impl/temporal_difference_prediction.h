#pragma once
#include "utility.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class TemporalDifferencePrediction
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef StateValueFunction<State_t> StateValueFunction_t;
	typedef PolicyFunction<State_t, Action_t> PolicyFunction_t;
	typedef paf::SharedPtr<StateValueFunction_t> StateValueFunctionPtr;
	typedef paf::SharedPtr<PolicyFunction_t> PolicyFunctionPtr;
	typedef paf::SharedPtr<TemporalDifferencePrediction> TemporalDifferencePredictionPtr;
public:
	TemporalDifferencePrediction(StateValueFunctionPtr stateValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f) :
		m_stateValueFunction(stateValueFunction),
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
	Action_t nextStep(float reward, const State_t& nextState)
	{
		float value = m_stateValueFunction.getValue(m_state);
		float target = reward + m_stateValueFunction.getValue(nextState) * m_discountRate;
		float newValue = value + (target - value) * m_learningRate;
		m_stateValueFunction.setValue(m_state, newValue);
		m_state = nextState;
		Action_t action = m_policy.takeAction(nextState);
		return action;
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		float value = m_stateValueFunction.getValue(m_state);
		float target = reward;
		if (!terminated)
		{
			target += m_stateValueFunction.getValue(nextState) * m_discountRate;
		}
		float newValue = value + (target - value) * m_learningRate;
		m_stateValueFunction.setValue(m_state, newValue);
	}
protected:
	StateValueFunctionPtr m_stateValueFunction;
	PolicyFunctionPtr m_policy;
	float m_learningRate;
	float m_discountRate;
public:
	static TemporalDifferencePredictionPtr Make(StateValueFunctionPtr stateValueFunction, PolicyFunctionPtr policy, float learningRate, float discountRate = 1.0f)
	{
		return TemporalDifferencePredictionPtr::Make(stateValueFunction, policy, learningRate, discountRate);
	}
};

END_RLTL_IMPL