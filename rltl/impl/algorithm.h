#pragma once
#include <vector>

namespace rltl
{
	//class Algorithm
	//{};

	template<typename Environment_t, typename Agent_t>
	class Train
	{
	public:
		void train(Environment_t& environment, Agent_t& agent, int32_t numEpisodes)
		{
			typedef Environment_t::State_t State_t;
			typedef Environment_t::Action_t Action_t;
			typedef Environment_t::Reward_t Reward_t;
			for (int32_t episode = 0; episode < numEpisodes; ++episode)
			{
				State_t state = environment.reset();
				Action_t action = agent.beginEpisode(state);
				while (true)
				{
					Reward_t reward;
					State_t nextState;
					EnvironmentStatus envStatus = environment.step(reward, nextState, action);
					action = agent.step(reward, nextState, envStatus != EnvironmentStatus::es_terminated);
					if (envStatus != EnvironmentStatus::es_normal)
					{
						agent.endEpisode();
						break;
					}
				}
			}
		}
	};



	template<typename State_t, typename Action_t, typename Reward_t, typename StateValueFunction_t>
	class MonteCarloPredication
	{
	public:
		Action_t beginEpisode(State_t& state)
		{
			m_state = state;
			return m_agent.takeAction(state);
		}
		Action_t step(const Reward_t& reward, const State_t& nextState, bool nonterminal)
		{
			SR sr;
			sr.state = m_state;
			sr.reward = reward;
			m_state = nextState;
			m_stateRewards.emplace_back(sr);
		}
		void endEpisode()
		{
			StateValueFunction_t::Value_t g = 0;
			size_t count = m_stateRewards.size();
			for (size_t i = 0; i < count; ++i)
			{
				SR& sr = m_stateRewards[count - 1 - i];
				g = g * m_discountRate + sr.reward;
				StateValueFunction_t::Value_t value = m_valueFunction.getValue(sr.state);
				StateValueFunction_t::Value_t newValue = value + (g - value) * m_learningRate;
				m_valueFunction.setValue(sr.state, newValue);
			}
		}
	public:
		struct SR
		{
			State_t state;
			Reward_t reward;
		};
		float m_discountRate;
		float m_learningRate;
		State_t m_state;
		std::vector<SR> m_stateRewards;
		StateValueFunction_t m_valueFunction;
		Agent_t& m_agent;
	};


	template<typename State_t, typename Action_t, typename Reward_t, typename StateValueFunction_t>
	class TemporalDifferencePredication
	{
	public:
		Action_t beginEpisode(State_t& state)
		{
			m_state = state;
			return m_valueFunction.takeAction(state);
		}
		Action_t step(const Reward_t& reward, const State_t& nextState, bool nonterminal)
		{
			SR sr;
			sr.state = m_state;
			sr.reward = reward;
			m_state = nextState;
			m_stateRewards.emplace_back(sr);
		}
		void endEpisode()
		{
			StateValueFunction_t::Value_t g = 0;
			size_t count = m_stateRewards.size();
			for (size_t i = 0; i < count; ++i)
			{
				SR& sr = m_stateRewards[count - 1 - i];
				g = g * m_discountRate + sr.reward;
				StateValueFunction_t::Value_t value = m_valueFunction.getValue(sr.state);
				StateValueFunction_t::Value_t newValue = value + (g - value) * m_learningRate;
				m_valueFunction.setValue(sr.state, newValue);
			}
		}
	public:
		StateValueFunction_t m_valueFunction;
	};

	template<typename State_t, typename Action_t, typename Reward_t, typename ActionValueFunction_t, typename Explorator_t>
	class QLearning
	{
	public:
		QLearning(ActionValueFunction_t& valueFunction, Explorator_t& explorator) :
			m_valueFunction(valueFunction),
	public:
		Action_t beginEpisode(State_t& firstState)
		{
			m_state = firstState;
			m_action = m_explorator.takeAction(m_state);
			return m_action;
		}
		Action_t step(const Reward_t& reward, const State_t& nextState, bool nonterminal)
		{
			ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state, m_action);
			ActionValueFunction_t::Value_t target = reward;
			if (nonterminal)
			{
				target += m_valueFunction.firstMaxValue(nextState);
			}
			ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
			m_valueFunction.setValue(m_state, m_action, newValue);
			if (nonterminal)
			{
				m_state = nextState;
				m_action = m_explorator.takeAction(m_state);
			}
			return m_action;
		}
		void endEpisode()
		{}
	public:
		Explorator_t& m_explorator;
		ActionValueFunction_t& m_valueFunction;
		State_t m_state;
		Action_t m_action;
		float m_learningRate;
	};


	template<typename State_t, typename Action_t, typename Reward_t, typename ActionValueFunction_t, typename Explorator_t>
	class Sarsa
	{
	public:
		QLearning(ActionValueFunction_t& valueFunction, Explorator_t& explorator) :
			m_explorator(explorator),
			m_valueFunction(valueFunction),
	public:
		Action_t beginEpisode(State_t& state)
		{
			m_state = state;
			m_action = m_explorator.takeAction(m_valueFunction, m_state);
			return m_action;
		}
		Action_t step(const Reward_t& reward, const State_t& nextState, bool nonterminal)
		{
			ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state, m_action);
			ActionValueFunction_t::Value_t target = reward;
			Action_t nextAction;
			if (nonterminal)
			{
				nextAction = m_explorator.takeAction(nextState);
				target += m_valueFunction.getValue(nextState, nextAction);
			}
			ActionValueFunction_t::Value_t newValue = m_value + (target - m_value) * m_learningRate;
			m_valueFunction.setValue(m_state, m_action, newValue);
			if (nonterminal)
			{
				m_state = nextState;
				m_action = nextAction;
			}
			return m_action;
		}
		void endEpisode()
		{}
	public:
		Explorator_t& m_explorator;
		ActionValueFunction_t& m_valueFunction;
		State_t m_state;
		Action_t m_action;
		float m_learningRate;
	};

	template<typename State_t, typename Action_t, typename Reward_t, typename ActionValueFunction_t, typename Explorator_t>
	class ExpectedSarsa
	{
	public:
		QLearning(ActionValueFunction_t& valueFunction, Explorator_t& explorator) :
			m_explorator(explorator),
			m_valueFunction(valueFunction),
	public:
		Action_t beginEpisode(State_t& state)
		{
			m_state = state;
			m_action = m_explorator.takeAction(m_valueFunction, m_state);
			return m_action;
		}
		Action_t step(const Reward_t& reward, const State_t& nextState, bool nonterminal)
		{
			ActionValueFunction_t::Value_t value = m_valueFunction.getValue(m_state, m_action);
			ActionValueFunction_t::Value_t target = reward;
			if (nonterminal)
			{
				target += m_explorator.getExpectedValue(nextState);
			}
			ActionValueFunction_t::Value_t newValue = value + (target - value) * m_learningRate;
			m_valueFunction.setValue(m_state, m_action, newValue);
			if (nonterminal)
			{
				m_state = nextState;
				m_action = m_explorator.takeAction(m_state);
			}
			return m_action;
		}
		void endEpisode()
		{}
	public:
		Explorator_t& m_explorator;
		ActionValueFunction_t& m_valueFunction;
		State_t m_state;
		Action_t m_action;
		float m_learningRate;
	};
}
