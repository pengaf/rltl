#include "cliff_walking.h"
#include <algorithm>

CliffWalking::CliffWalking()
{
	m_stateSpace = ConcreteStateSpacePtr::Make(4*12);
	m_actionSpace = ConcreteActionSpacePtr::Make(4);

	m_height = 4;
	m_width = 12;
	m_startX = 0;
	m_startY = 0;
	m_endX = 11;
	m_endY = 0;
	m_currentX = m_startX;
	m_currentY = m_startY;
}

CliffWalking::StateSpacePtr CliffWalking::stateSpace()
{
	return m_stateSpace;
}

CliffWalking::ActionSpacePtr CliffWalking::actionSpace()
{
	return m_actionSpace;
}

CliffWalking::State_t CliffWalking::reset(int seed)
{
	m_currentX = m_startX;
	m_currentY = m_startY;
	return m_currentX + m_currentY * m_width;
}

rltl::impl::EnvironmentStatus CliffWalking::step(float& reward, State_t& nextState, const Action_t& action)
{
	switch (action)
	{
	case 0://down
		m_currentY = std::max(m_currentY - 1, 0);
		break;
	case 1://up
		m_currentY = std::min(m_currentY + 1, m_height - 1);
		break;
	case 2://left
		m_currentX = std::max(m_currentX - 1, 0);
		break;
	case 3://right
		m_currentX = std::min(m_currentX + 1, m_width - 1);
		break;
	}
	reward = -1;
	if (m_currentY == 0 && m_currentX > 0 && m_currentX < m_width - 1)
	{
		reward = -100;
		m_currentX = m_startX;
		m_currentY = m_startY;
	}
	nextState = m_currentX + m_currentY * m_width;
	if (m_currentY == 0 && m_currentX == m_width - 1)
	{
		return rltl::impl::EnvironmentStatus::es_terminated;
	}
	return rltl::impl::EnvironmentStatus::es_normal;
}

CliffWalking2::CliffWalking2()
{
	State_t low = { 0, 0};
	State_t high = { 4, 12 };

	m_stateSpace = ConcreteStateSpacePtr::Make(low, high);
	m_actionSpace = ConcreteActionSpacePtr::Make(4);


	m_startState = { 0,0 };
	m_endState = { 0,11 };
	m_currentState = m_startState;
}

CliffWalking2::StateSpacePtr CliffWalking2::stateSpace()
{
	return m_stateSpace;
}

CliffWalking2::ActionSpacePtr CliffWalking2::actionSpace()
{
	return m_actionSpace;
}

CliffWalking2::State_t CliffWalking2::reset(int seed)
{
	m_currentState = m_startState;

	return m_currentState;
}

rltl::impl::EnvironmentStatus CliffWalking2::step(float& reward, State_t& nextState, const Action_t& action)
{
	switch (action)
	{
	case 0://down
		m_currentState[0] = std::max(m_currentState[0] - 1, m_stateSpace->low()[0]);
		break;
	case 1://up
		m_currentState[0] = std::min(m_currentState[0] + 1, m_stateSpace->high()[0] - 1);
		break;
	case 2://left
		m_currentState[1] = std::max(m_currentState[1] - 1, m_stateSpace->low()[1]);
		break;
	case 3://right
		m_currentState[1] = std::min(m_currentState[1] + 1, m_stateSpace->high()[1] - 1);
		break;
	}
	reward = -1;
	if (m_currentState[0] == 0 && m_currentState[1] > m_stateSpace->low()[1] && m_currentState[1] < m_stateSpace->high()[1] - 1)
	{
		reward = -100;
		m_currentState = m_startState;
	}
	nextState = m_currentState;
	if (m_currentState == m_endState)
	{
		return rltl::impl::EnvironmentStatus::es_terminated;
	}
	return rltl::impl::EnvironmentStatus::es_normal;
}