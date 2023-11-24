#pragma once
#include "utility.h"
#include "../math/random.h"
#include <assert.h>
#include <vector>

BEGIN_RLTL_IMPL

template<typename ActionValueFunction_t, bool random_select_max_action=false>
class EpsilonGreedy
{
public:
	typedef typename ActionValueFunction_t::State_t State_t;
	typedef typename ActionValueFunction_t::Action_t Action_t;
	typedef typename ActionValueFunction_t::Value_t Value_t;
public:
	EpsilonGreedy(ActionValueFunction_t& valueFunction, float epsilon = 1.0) :
		m_valueFunction(valueFunction),
		m_actionCount(valueFunction.actionCount()),
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
			if constexpr (random_select_max_action)
			{
				return m_valueFunction.randomMaxAction(state);
			}
			else
			{
				return m_valueFunction.firstMaxAction(state);
			}
		}
	}
	Value_t getExpectedValue(const State_t& state)
	{
		return m_valueFunction.meanValue(state) * m_epsilon + m_valueFunction.firstMaxValue(state)*(1 - m_epsilon);
	}
protected:
	ActionValueFunction_t& m_valueFunction;
	uint32_t m_actionCount;
	float m_epsilon;
public:
};

template<typename ActionValueFunction_t>
inline static EpsilonGreedy<ActionValueFunction_t> MakeEpsilonGreedy(ActionValueFunction_t& valueFunction, float epsilon = 1.0)
{
	return EpsilonGreedy(valueFunction, epsilon);
}

template<typename ActionValueFunction_t, bool random_select_max_action = false>
class LinearDecayEpsilonGreedy
{
public:
	typedef typename ActionValueFunction_t::State_t State_t;
	typedef typename ActionValueFunction_t::Action_t Action_t;
	typedef typename ActionValueFunction_t::Value_t Value_t;
public:
	LinearDecayEpsilonGreedy(ActionValueFunction_t& valueFunction, float startEpsilon, float endEpsilon, uint32_t decaySteps) :
		m_valueFunction(valueFunction),
		m_actionCount(valueFunction.actionCount()),
		m_epsilon(startEpsilon),
		m_startEpsilon(startEpsilon),
		m_endEpsilon(endEpsilon),
		m_decaySteps(decaySteps),
		m_steps(0)
	{}
public:
	float getEpsilon(float epsilon) const
	{
		return m_epsilon;
	}
	Action_t takeAction(const State_t& state)
	{
		if (rltl::math::Random::rand() < m_epsilon)
		{
			return rltl::math::Random::randint() % m_actionCount;
		}
		else
		{
			if constexpr (random_select_max_action)
			{
				return m_valueFunction.randomMaxAction(state);
			}
			else
			{
				return m_valueFunction.firstMaxAction(state);
			}
		}
		if (m_steps < m_decaySteps)
		{
			++m_steps;
			m_epsilon = m_startEpsilon + (m_endEpsilon - m_startEpsilon)*double(m_steps)/double(m_decaySteps);
		}
	}
	Value_t getExpectedValue(const State_t& state)
	{
		return m_valueFunction.meanValue() * m_epsilon + m_valueFunction.firstMaxValue()*(1 - m_epsilon);
	}
protected:
	ActionValueFunction_t& m_valueFunction;
	uint32_t m_actionCount;
	float m_epsilon;
	float m_startEpsilon;
	float m_endEpsilon;
	uint32_t m_decaySteps;
	uint32_t m_steps;
};

END_RLTL_IMPL