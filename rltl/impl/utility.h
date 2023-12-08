#pragma once
#include <stdint.h>
#include <assert.h>

#define BEGIN_RLTL_IMPL	namespace rltl { namespace impl {
#define END_RLTL_IMPL	} }

BEGIN_RLTL_IMPL

enum class TargetEvaluationMethod
{
	q_learning,
	sarsa,
	expected_sarsa
};

enum class SampleDepthCategory
{
	td_0,
	n_step_td,
	monte_carlo,
};

END_RLTL_IMPL