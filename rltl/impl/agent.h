#pragma once
#include "utility.h"
#include <stdint.h>
#include <assert.h>
#include <vector>
#include "../arg.h"

BEGIN_RLTL_IMPL

//struct AgentOptions
//{
//	AgentOptions(float learningRate, float discountRate) :
//		m_learningRate(learningRate),
//		m_discountRate(discountRate)
//	{}
//	RLTL_ARG(float, learningRate);
//	RLTL_ARG(float, discountRate);
//};
//
//class Agent
//{
//public:
//	template<typename Environment_t, typename ValueFunction_t>
//	void initialize(Environment_t& environment, ValueFunction_t& valueFunction)
//	{
//	}
//public:
//	template<typename Algorithm_t>
//	void train(Algorithm_t& algorithm, int32_t numEpisodes)
//	{
//		typedef Environment_t::State_t State_t;
//		typedef Environment_t::Action_t Action_t;
//		for (int32_t episode = 0; episode < numEpisodes; ++episode)
//		{
//			State_t state = environment.reset();
//			Action_t action = algorithm.firstStep(state);
//			while (true)
//			{
//				float reward;
//				State_t nextState;
//				EnvironmentStatus envStatus = environment.step(reward, nextState, action);
//				if (EnvironmentStatus::es_normal == envStatus)
//				{
//					action = algorithm.nextStep(reward, nextState);
//				}
//				else
//				{
//					algorithm.lastStep(reward, nextState, envStatus == EnvironmentStatus::es_terminated);
//					break;
//				}
//			}
//		}
//	}
//};


END_RLTL_IMPL
