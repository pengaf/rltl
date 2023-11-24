#pragma once
#include "utility.h"
#include "space.h"
#include "space_normalizer.h"

BEGIN_RLTL_IMPL

enum class EnvironmentStatus
{
	es_normal,
	es_terminated,
	es_truncated
};

template<typename StateSpace_t, typename ActionSpace_t, typename Reward_t = float>
class Environment
{
public:
	Environment(const StateSpace_t& stateSpace, const ActionSpace_t& actionSpace) :
		m_stateSpace(stateSpace),
		m_actionSpace(actionSpace)
	{}
public:
	typedef StateSpace_t StateSpace_t;
	typedef ActionSpace_t ActionSpace_t;
	typedef Reward_t Reward_t;
	typedef typename StateSpace_t::Element_t State_t;
	typedef typename ActionSpace_t::Element_t Action_t;
public:
	const StateSpace_t& stateSpace() const
	{
		return m_stateSpace;
	}
	const ActionSpace_t& actionSpace() const
	{
		return m_actionSpace;
	}
protected:
	const StateSpace_t& m_stateSpace;
	const ActionSpace_t& m_actionSpace;
	//concept
	//State_t reset(int seed = 0);
	//EnvironmentStatus step(Reward_t& reward, State_t& nextState, const Action_t& action);
	//void close();
};

template<typename Environment_t, typename DiscreteState_t = uint32_t, typename DiscreteAction_t = uint32_t>
class DiscreteEnvironment
{
public:
	typedef typename ToNormalizedDiscrete<typename Environment_t::StateSpace_t, DiscreteState_t>::Normalizer_t StateSpaceNormalizer_t;
	typedef typename ToNormalizedDiscrete<typename Environment_t::ActionSpace_t, DiscreteAction_t>::Normalizer_t ActionSpaceNormalizer_t;
	typedef typename StateSpaceNormalizer_t::DstSpace_t StateSpace_t;
	typedef typename ActionSpaceNormalizer_t::DstSpace_t ActionSpace_t;
	typedef typename Environment_t::Reward_t Reward_t;
	typedef typename StateSpace_t::Element_t State_t;
	typedef typename ActionSpace_t::Element_t Action_t;
public:
	DiscreteEnvironment(Environment_t& environment) :
		m_environment(environment),
		m_stateSpaceNormalizer(environment.stateSpace()),
		m_actionSpaceNormalizer(environment.actionSpace())
	{}
	DiscreteEnvironment(Environment_t& environment, size_t discreteCount) :
		m_environment(environment),
		m_stateSpaceNormalizer(environment.stateSpace(), discreteCount),
		m_actionSpaceNormalizer(environment.actionSpace())
	{}
public:
	const StateSpace_t& stateSpace() const
	{
		return m_stateSpaceNormalizer.dstSpace();
	}
	const ActionSpace_t& actionSpace() const
	{
		return m_actionSpaceNormalizer.dstSpace();
	}
public:
	State_t reset(int seed = 0)
	{
		Environment_t::State_t originState = m_environment.reset(seed);
		State_t state = m_stateSpaceNormalizer.normalize(originState);
		return state;
	}
	EnvironmentStatus step(Reward_t& reward, State_t& nextState, const Action_t& action)
	{
		Environment_t::Action_t originAction = m_actionSpaceNormalizer.restore(action);
		Environment_t::State_t originNextState;
		EnvironmentStatus status = m_environment.step(reward, originNextState, originAction);
		nextState = m_stateSpaceNormalizer.normalize(originNextState);
		return status;
	}
	void close()
	{
		m_environment.close();
	}
protected:
	Environment_t& m_environment;
	StateSpaceNormalizer_t m_stateSpaceNormalizer;
	ActionSpaceNormalizer_t m_actionSpaceNormalizer;
};

END_RLTL_IMPL
