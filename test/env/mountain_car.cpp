#include "mountain_car.h"
#include "../../rltl/math/random.h"
#include <algorithm>
#include <math.h>

MountainCar::MountainCar(uint32_t maxStep) :
	Environment(m_stateSpace, m_actionSpace),
	m_stateSpace({ -1.2, -0.07}, { 0.5, 0.07}),
	m_actionSpace(0, 3),
	m_maxStep(maxStep),
	m_step(0)
{
	m_currentState = { 0.6, 0 };
}

MountainCar::State_t MountainCar::reset(int seed)
{
	m_step = 0;
	m_currentState = { rltl::math::Random::rand()*0.2f - 0.6f, 0.0f };
	return m_currentState;
}

rltl::impl::EnvironmentStatus MountainCar::step(Reward_t& reward, State_t& nextState, const Action_t& action)
{
	float px = m_currentState[0];
	float vx = m_currentState[1];
	float acc = (action - 1)*0.001;
	vx = std::clamp<float>(vx + acc - cos(px*3.0) *0.0025, m_stateSpace.low(1), m_stateSpace.high(1));
	px = std::clamp<float>(px + vx, m_stateSpace.low(0), m_stateSpace.high(0));
	if (px == m_stateSpace.low(0))
	{
		vx = 0;
	}
	bool terminated = px == m_stateSpace.high(0);
	m_currentState = { px, vx };
	nextState = m_currentState;
	reward = -1.0f;

	if (terminated)
	{
		return rltl::impl::EnvironmentStatus::es_terminated;
	}
	++m_step;
	if (m_step >= m_maxStep)
	{
		return rltl::impl::EnvironmentStatus::es_truncated;
	}
	return rltl::impl::EnvironmentStatus::es_normal;
}