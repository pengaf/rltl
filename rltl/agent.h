#pragma once
#include <string>
#include <vector>
#include "data_type.h"

namespace rltl
{

	template<typename State_t, typename Action_t>
	class Agent
	{
	public:
		typedef State_t State_t;
		typedef Action_t Action_t;
		
	public:
		//State_t reset(int seed = 0) = 0;
		//EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action) = 0;
		//void close() = 0;
		Action_t beginEpisode(State_t& state)
		Action_t step(float& reward, State_t& nextState, bool terminated);
		void endEpisode();
	};


	template<typename StateValueFunction_t, typename Action_t>
	class StateValueAgent
	{
	public:
		typedef StateValueFunction_t StateValueFunction_t;
		typedef typename StateValueFunction_t::State_t State_t;
		typedef Action_t Action_t;
	public:
		Action_t takeAction(const State_t& state)
		{
			m_valueFunction.
		}
	public:
		StateValueFunction_t m_valueFunction;
	};

	template<typename State_t, typename Action_t>
	class ControlAgent
	{
	public:
		Action_t takeAction(const State_t& state);
	};

	class ActionValueAgent
	{

	};
}