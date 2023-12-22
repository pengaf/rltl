#include "cart_pole.h"
#include "../../rltl/math/random.h"
#include <algorithm>
#include <math.h>

const float c_gravity = 9.8;
const float c_masscart = 1.0;
const float c_masspole = 0.1;
const float c_total_mass = c_masspole + c_masscart;
const float c_length = 0.5;//  # actually half the pole's length
const float c_polemass_length = c_masspole * c_length;
const float c_force_mag = 10.0;
const float c_tau = 0.02;//  # seconds between state updates
const float c_theta_threshold_radians = 12 * 2 * 3.1415926 / 360;
const float c_x_threshold = 2.4;

CartPole::CartPole(uint32_t maxStep):
	m_maxStep(maxStep),
	m_step(0)
{
	State_t low = { -c_x_threshold * 2, -FLT_MAX, -c_theta_threshold_radians * 2, -FLT_MAX };
	State_t high = { c_x_threshold * 2, FLT_MAX, c_theta_threshold_radians * 2, FLT_MAX };

	m_stateSpace = ConcreteStateSpacePtr::Make(low, high);
	m_actionSpace = ConcreteActionSpacePtr::Make(2);
	m_currentState = { 0, 0 , 0 , 0 };
}

CartPole::StateSpacePtr CartPole::stateSpace()
{
	return m_stateSpace;
}

CartPole::ActionSpacePtr CartPole::actionSpace()
{
	return m_actionSpace;
}

CartPole::State_t CartPole::reset(int seed)
{
	m_step = 0;
	m_currentState = { 
		rltl::math::Random::rand()*0.1f - 0.05f,
		rltl::math::Random::rand()*0.1f - 0.05f,
		rltl::math::Random::rand()*0.1f - 0.05f,
		rltl::math::Random::rand()*0.1f - 0.05f,
	};
	return m_currentState;
}

rltl::impl::EnvironmentStatus CartPole::step(float& reward, State_t& nextState, const Action_t& action)
{
	float x = m_currentState[0];
	float x_dot = m_currentState[1];
	float theta = m_currentState[2];
	float theta_dot = m_currentState[3];

	float force = action == 1 ? c_force_mag : -c_force_mag;
	float costheta = std::cos(theta);
	float sintheta = std::sin(theta);
	float temp = (force + c_polemass_length * theta_dot * theta_dot * sintheta) / c_total_mass;
	float thetaacc = (c_gravity * sintheta - costheta * temp) / (c_length * (4.0 / 3.0 - c_masspole * costheta*costheta / c_total_mass));
	float xacc = temp - c_polemass_length * thetaacc * costheta / c_total_mass;

	x = x + c_tau * x_dot;
	x_dot = x_dot + c_tau * xacc;
	theta = theta + c_tau * theta_dot;
	theta_dot = theta_dot + c_tau * thetaacc;

	m_currentState = { x, x_dot, theta, theta_dot };

	bool terminated = (x < -c_x_threshold || x > c_x_threshold || theta < -c_theta_threshold_radians || theta > c_theta_threshold_radians);

	nextState = m_currentState;
	reward = terminated ? 0 : 1.0f;

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