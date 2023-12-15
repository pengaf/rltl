#pragma once
#include "utility.h"
#include "../math/random.h"
#include "callback.h"

BEGIN_RLTL_IMPL

template<typename State_t, typename Action_t>
class GreedyAction : public DiscretePolicyFunction<State_t, Action_t>
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef ActionValueFunction<State_t, Action_t> ActionValueFunction_t;
	typedef paf::SharedPtr<ActionValueFunction_t> ActionValueFunctionPtr;
	typedef paf::SharedPtr<GreedyAction> GreedyActionPtr;
public:
	GreedyAction(ActionValueFunctionPtr actionValueFunction, bool firstMax) :
		m_actionValueFunction(actionValueFunction),
		m_firstMax(firstMax)
	{}
public:
	Action_t takeAction(const State_t& state) const override
	{
		return m_actionValueFunction->maxAction(state, m_firstMax);
	}
	uint32_t actionCount() const override
	{
		return m_actionValueFunction->actionCount();
	}
protected:
	ActionValueFunctionPtr m_actionValueFunction;
	bool m_firstMax;
public:
	static GreedyActionPtr Make(ActionValueFunctionPtr actionValueFunction, bool firstMax = true)
	{
		return GreedyActionPtr::Make(actionValueFunction, firstMax);
	}
};

template<typename State_t, typename Action_t>
class EpsilonGreedy : public DiscretePolicyFunction<State_t, Action_t>
{
public:
	typedef State_t State_t;
	typedef Action_t Action_t;
	typedef DiscretePolicyFunction<State_t, Action_t> DiscretePolicyFunction_t;
	typedef paf::SharedPtr<DiscretePolicyFunction_t> DiscretePolicyFunctionPtr;
	typedef paf::SharedPtr<EpsilonGreedy> EpsilonGreedyPtr;
public:
	EpsilonGreedy(DiscretePolicyFunctionPtr policy, float epsilon) :
		m_policy(policy),
		m_actionCount(policy->actionCount()),
		m_epsilon(epsilon)
	{}
public:
	float getEpsilon(float epsilon) const
	{
		return m_epsilon;
	}
	void setEpsilon(float epsilon)
	{
		m_epsilon = epsilon;
	}
	Action_t takeAction(const State_t& state)
	{
		if (rltl::math::Random::rand() < m_epsilon)
		{
			return rltl::math::Random::randint(m_actionCount) % m_actionCount;
		}
		else
		{
			return m_policy->takeAction(state);
		}
	}
	//expected sarsa
	float getExpectedValue(std::vector<float>& actionValues)
	{
		size_t count = actionValues.size();
		float sumValue, maxValue;
		sumValue = maxValue = actionValues[0];
		for (size_t i = 1; i < count; ++i)
		{
			float value = actionValues[i];
			sumValue += value;
			if (maxValue < value)
			{
				maxValue = value;
			}
		}
		return sumValue * m_epsilon / float(count) + maxValue * (1.0f - m_epsilon);
	}
protected:
	DiscretePolicyFunctionPtr m_policy;
	uint32_t m_actionCount;
	float m_epsilon;
public:
	static EpsilonGreedyPtr Make(DiscretePolicyFunctionPtr policy, float epsilon = 1.0)
	{
		return EpsilonGreedyPtr::Make(policy, epsilon);
	}
};


template<typename State_t, typename Action_t>
class EpsilonGreedyLinearDecay : public Callback
{
public:
	typedef EpsilonGreedy<State_t, Action_t> EpsilonGreedy_t;
	typedef paf::SharedPtr<EpsilonGreedy_t> EpsilonGreedyPtr;
	typedef paf::SharedPtr<EpsilonGreedyLinearDecay> EpsilonGreedyLinearDecayPtr;
public:
	EpsilonGreedyLinearDecayPtr(EpsilonGreedyPtr epsilonGreedy, float startEpsilon, float endEpsilon, uint32_t decaySteps):
		m_epsilonGreedy(epsilonGreedy),
		m_startEpsilon(startEpsilon),
		m_endEpsilon(endEpsilon),
		m_decaySteps(decaySteps)
	{}
	void beginTrain(uint32_t numEpisodes)
	{
		m_accStep = 0;
	}
	void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step)
	{
		if(m_accStep < m_decaySteps)
		{
			++m_accStep;
			float epsilon = m_startEpsilon + (m_endEpsilon - m_startEpsilon)*(m_accStep) / (m_decaySteps);
			m_epsilonGreedy.setEpsilon(epsilon);
		}
	}
protected:
	EpsilonGreedyPtr m_epsilonGreedy;
	float m_startEpsilon;
	float m_endEpsilon;
	uint32_t m_decaySteps;
	uint32_t m_accStep;
public:
	static EpsilonGreedyLinearDecayPtr Make(EpsilonGreedyPtr epsilonGreedy, float startEpsilon, float endEpsilon, uint32_t decaySteps)
	{
		return EpsilonGreedyLinearDecayPtr::Make(epsilonGreedy, startEpsilon, endEpsilon, decaySteps);
	}
};

END_RLTL_IMPL