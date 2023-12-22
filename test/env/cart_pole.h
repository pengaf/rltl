#pragma once
#include "../../rltl/impl/environment.h"

class CartPole : public rltl::impl::Environment<rltl::impl::Array<float, 4>, uint32_t>
{
public:
	typedef rltl::impl::VectorSpace<State_t> ConcreteStateSpace_t;
	typedef rltl::impl::IndexSpace<Action_t> ConcreteActionSpace_t;
	typedef paf::SharedPtr<ConcreteStateSpace_t> ConcreteStateSpacePtr;
	typedef paf::SharedPtr<ConcreteActionSpace_t> ConcreteActionSpacePtr;
public:
	CartPole(uint32_t maxStep = 200);
public:
	StateSpacePtr stateSpace();
	ActionSpacePtr actionSpace();
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action);
	void close() {}
private:
	ConcreteStateSpacePtr m_stateSpace;
	ConcreteActionSpacePtr m_actionSpace;
	State_t m_currentState;
	uint32_t m_maxStep;
	uint32_t m_step;
};
