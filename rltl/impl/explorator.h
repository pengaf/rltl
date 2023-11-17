#pragma once
#include "utility.h"
#include "../math/random.h"
#include <assert.h>
#include <vector>

BEGIN_RLTL_IMPL

template<typename ActionValueFunction_t>
class EpsilonGreedy
{
public:
	typedef typename ActionValueFunction_t::Action_t Action_t;
	typedef typename ActionValueFunction_t::Value_t Value_t;
public:
	EpsilonGreedy(ActionValueFunction_t& valueFunction, float epsilon = 1.0) :
		m_valueFunction(valueFunction),
		m_actionCount(valueFunction.actionCount()),
		m_epsilon(epsilon)
	{}
public:
	Action_t takeAction(const State_t& state)
	{
		if (rltl::math::Random::rand() < m_epsilon)
		{
			return rltl::math::Random::randint() % m_actionCount;
		}
		else
		{
			return m_valueFunction.firstMaxAction(state);
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
};

END_RLTL_IMPL