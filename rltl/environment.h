#pragma once
#include <string>
#include <vector>
#include "data_type.h"

namespace rltl
{
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
		typedef State_t State_t;
		typedef Action_t Action_t;
		typedef Reward_t Reward_t;
	public:
		virtual State_t reset(int seed = 0) = 0;
		virtual EnvironmentStatus step(Reward_t& reward, State_t& nextState, const Action_t& action) = 0;
		virtual void close() = 0;
    };
}