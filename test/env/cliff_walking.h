#pragma once
#include "../../rltl/impl/environment.h"

class CliffWalking : public rltl::impl::Environment<uint32_t, uint32_t>
{
public:
	CliffWalking();
public:
	StateSpacePtr stateSpace();
	ActionSpacePtr actionSpace();
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action);
	void close(){}
private:
	typedef rltl::impl::IndexSpace<State_t> ConcreteStateSpace_t;
	typedef rltl::impl::IndexSpace<Action_t> ConcreteActionSpace_t;
	typedef paf::SharedPtr<ConcreteStateSpace_t> ConcreteStateSpacePtr;
	typedef paf::SharedPtr<ConcreteActionSpace_t> ConcreteActionSpacePtr;
private:
	ConcreteStateSpacePtr m_stateSpace;
	ConcreteActionSpacePtr m_actionSpace;
	int m_height;
	int m_width;
	int m_startX;
	int m_startY;
	int m_endX;
	int m_endY;
	int m_currentX;
	int m_currentY;
};

class CliffWalking2 : public rltl::impl::Environment<rltl::impl::Array<int32_t, 2>, uint32_t>
{
public:
	CliffWalking2();
public:
	StateSpacePtr stateSpace();
	ActionSpacePtr actionSpace();
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action);
	void close() {}
private:
	typedef rltl::impl::VectorSpace<State_t> ConcreteStateSpace_t;
	typedef rltl::impl::IndexSpace<Action_t> ConcreteActionSpace_t;
	typedef paf::SharedPtr<ConcreteStateSpace_t> ConcreteStateSpacePtr;
	typedef paf::SharedPtr<ConcreteActionSpace_t> ConcreteActionSpacePtr;
private:
	ConcreteStateSpacePtr m_stateSpace;
	ConcreteActionSpacePtr m_actionSpace;
	//int m_height;
	//int m_width;
	State_t m_startState;
	State_t m_endState;
	State_t m_currentState;
};
