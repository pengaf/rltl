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

	template<typename StateSpace_t, typename ActionSpace_t>
    class Environment
    {
	public:
		typedef State_t State_t;
		typedef Action_t Action_t;
		
	public:
		virtual State_t reset(int seed = 0) = 0;
		virtual EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action) = 0;
		virtual void close() = 0;
    };
}