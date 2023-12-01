#pragma once
#include "../../rltl/impl/environment.h"

class MountainCar : public rltl::impl::Environment<rltl::impl::BoxSpace<2>, rltl::impl::DiscreteSpace<>>
{
public:
	MountainCar(uint32_t maxStep = 500);
public:
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(Reward_t& reward, State_t& nextState, const Action_t& action);
	void close() {}
private:
	StateSpace_t m_stateSpace;
	ActionSpace_t m_actionSpace;
	State_t m_currentState;
	uint32_t m_maxStep;
	uint32_t m_step;
};
