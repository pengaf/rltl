#pragma once
#include "utility.h"
#include <stdlib.h>
#include <random>

BEGIN_RLTL_IMPL

class Random
{
public:
	static float rand()
	{
		std::uniform_real_distribution<float> distribution(0, 1.0);
		return distribution(generator());
		//return ::rand() / float(RAND_MAX);
	}
	static int randint(int high)
	{
		std::uniform_int_distribution<int> distribution(0, high - 1);
		return distribution(generator());
		//return ::rand() % high;
	}
	static std::default_random_engine& generator()
	{
		static std::default_random_engine s_generator;
		return s_generator;
	}
};

END_RLTL_IMPL
