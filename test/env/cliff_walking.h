#pragma once
#include "../../rltl/impl/environment.h"

class CliffWalking : public rltl::impl::Environment<rltl::impl::NormalizedDiscreteSpace<>, rltl::impl::NormalizedDiscreteSpace<>>
{
public:
	CliffWalking();
public:
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action);
	void close(){}
private:
	StateSpace_t m_stateSpace;
	ActionSpace_t m_actionSpace;
	int m_height;
	int m_width;
	int m_startX;
	int m_startY;
	int m_endX;
	int m_endY;
	int m_currentX;
	int m_currentY;
};

class CliffWalking2 : public rltl::impl::Environment<rltl::impl::MultiDiscreteSpace<2>, rltl::impl::DiscreteSpace<>>
{
public:
	CliffWalking2();
public:
	State_t reset(int seed = 0);
	rltl::impl::EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action);
	void close() {}
private:
	StateSpace_t m_stateSpace;
	ActionSpace_t m_actionSpace;
	//int m_height;
	//int m_width;
	State_t m_startState;
	State_t m_endState;
	State_t m_currentState;
};
