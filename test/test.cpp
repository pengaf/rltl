#include <iostream>
#include "../rltl/impl/action_value_table.h"
#include "../rltl/impl/exploration.h"
#include "../rltl/impl/sarsa.h"
#include "../rltl/impl/expected_sarsa.h"
#include "../rltl/impl/q_learning.h"
#include "../rltl/impl/deep_q_network.h"
#include "../rltl/impl/deep_reinforce.h"
#include "../rltl/impl/deep_actor_critic.h"
#include "../rltl/impl/deep_actor_critic2.h"

#include "../rltl/impl/action_value_net.h"
#include "../rltl/impl/state_value_net.h"
#include "../rltl/impl/policy_net.h"

#include "../rltl/impl/algorithm.h"
#include "../rltl/impl/trainer.h"
#include "../rltl/impl/callback.h"
#include "env/cliff_walking.h"
#include "env/mountain_car.h"
#include "env/cart_pole.h"
#include <chrono>

template<typename Agent_t>
class RewardStat
{
public:
	typedef typename Agent_t::State_t State_t;
	typedef typename Agent_t::Action_t Action_t;
public:
	RewardStat(Agent_t& agent) :
		m_agent(agent)
	{}
	Action_t firstStep(State_t& firstState)
	{
		m_totalReward = 0;
		return m_agent.firstStep(firstState);
	}
	Action_t nextStep(float reward, const State_t& nextState)
	{
		m_totalReward += reward;
		return m_agent.nextStep(reward, nextState);
	}
	void lastStep(float reward, const State_t& nextState, bool terminated)
	{
		m_totalReward += reward;
		m_rewardStat.push_back(m_totalReward);
		return m_agent.lastStep(reward, nextState, terminated);
	}
public:
	Agent_t& m_agent;
	float m_totalReward;
	std::vector<float> m_rewardStat;
};

class RewardStat2 : public rltl::impl::Callback
{
public:
	RewardStat2(uint32_t numEpisodes)
	{
		m_numEpisodes = numEpisodes;
	}
	
	void beginTrain() override
	{
		m_rewardStat.clear();
		m_rewardStat.reserve(m_numEpisodes);
		m_maxReward = -FLT_MAX;
		m_allSteps = 0;
		m_start = std::chrono::high_resolution_clock::now();
	}
	void beginEpisode(uint32_t episode)  override
	{}
	void endEpisode(uint32_t episode, uint32_t totalStep, float totalReward) override
	{
		m_allSteps += totalStep;
		m_rewardStat.push_back(totalReward);
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration duration = end - m_start;
		float seconds = duration.count()*0.000000001;
		uint32_t stepPerSecond = m_allSteps / seconds;
		bool newMaxReward = m_maxReward < totalReward;
		if(newMaxReward)
		{
			m_maxReward = totalReward;
		}
		if (newMaxReward || episode * 10 % m_numEpisodes == 0)
		{
			printf("episode:%d,step:%d,reward:%f,time:%f,step/second: %d\n", episode, totalStep, m_rewardStat.back(), seconds, stepPerSecond);
		}
	}
	void endTrain()  override
	{
		std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration duration = end - m_start;
		float seconds = duration.count()*0.000000001;
		uint32_t stepPerSecond = m_allSteps / seconds;
		printf("last:%f, max:%f, step:%d, time:%f, step/second: %d\n", m_rewardStat.back(), m_maxReward, m_allSteps, seconds, stepPerSecond);
	}
public:
	std::vector<float> m_rewardStat;
	float m_maxReward;
	std::chrono::high_resolution_clock::time_point m_start;
	uint32_t m_allSteps;
	uint32_t m_numEpisodes;
};

//class TestStepCallback : public rltl::impl::Callback
//{
//public:
//	virtual void beginStep(uint32_t numEpisodes, uint32_t episode, uint32_t step) override
//	{
//		++m_totalStep;
//	}
//	virtual void endTrain(uint32_t numEpisodes) override
//	{
//		std::cout << m_totalStep << std::endl;
//	}
//	uint32_t m_totalStep{ 0 };
//};


//void test_table()
//{
//	//CliffWalking2 env2;
//	//rltl::impl::DiscreteEnvironment<CliffWalking2> env(env2);
//	//CliffWalking env;
//	MountainCar env2; rltl::impl::DiscreteEnvironment<MountainCar> env(env2,10);
//
//
//	rltl::impl::ActionValueTable<> actionValueTable(env.stateSpace().count(), env.actionSpace().count());
//	auto greedyAction = rltl::impl::MakeGreedyAction(actionValueTable);
//	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(greedyAction, 0.1f);
//	//auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actionValueFunction, 0.1f);
//
//	//auto agent = rltl::impl::MakeSarsa(actionValueTable, epsilonGreedy, 0.1f);
//	//auto agent = rltl::impl::MakeExpectedSarsa(actionValueTable, epsilonGreedy, 0.1f);
//	auto agent = rltl::impl::MakeQLearning(actionValueTable, epsilonGreedy, 0.1f);
//	//RewardStat<decltype(agent)> rewardStat(agent);
//
//	RewardStat2 callback;
//	rltl::impl::train(env, agent, 5000, &callback);
//}

void test_dqn(const std::string& filename)
{
	torch::Device device(torch::DeviceType::CUDA);
	//rltl::impl::Callback* stepCallback = new TestStepCallback;
	typedef MountainCar Env;
	//typedef CartPole Env;
	
	auto env = paf::SharedPtr<Env>::Make();
	//Env env;
	Env::ConcreteStateSpacePtr stateSpacePtr = env->stateSpace();
	Env::ConcreteActionSpacePtr actionSpacePtr = env->actionSpace();

	typedef rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> ActionValueNet;

	auto actionValueNet = rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t>::Make(rltl::impl::Vector_dimension(stateSpacePtr->low()), actionSpacePtr->count(), 128, 1, false);

	//try
	//{
	//	rltl::impl::NN_loadModule(actionValueNet->module(), filename);
	//}
	//catch (const std::exception& e)
	//{
	//	std::cout << e.what();
	//}

	actionValueNet->get()->device(device);
	std::shared_ptr<torch::optim::AdamW> optimizer(new torch::optim::AdamW((*actionValueNet)->parameters(), torch::optim::AdamWOptions(1e-3)));

	auto greedyAction = rltl::impl::GreedyAction<Env::State_t, Env::Action_t>::Make(actionValueNet);
	auto epsilonGreedy = rltl::impl::EpsilonGreedy<Env::State_t, Env::Action_t>::Make(greedyAction, 0.1f);
	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
	//rltl::impl::DeepQLearningOptions options(0.98, 500, 64, 5, 5, false);
	rltl::impl::DeepQLearningOptions options(0.98, 64, false);
	options.targetNetwork(5);
	options.experienceReplay(10000, 500, 5);
	//options.prioritizedExperienceReplay(10000, 500, 5, 0.2, 0.5);
	//options.multiStep(4, true);

	//options.multiStep(4);// .multiStepCompound(true);

	//rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
	auto agent = rltl::impl::MakeDQN(actionValueNet, optimizer, epsilonGreedy, options);
	uint32_t numEpisodes = 1500;
	RewardStat2 rewardStat(numEpisodes);
	rltl::impl::Trainer<Env::State_t, Env::Action_t>::TrainEpisodes(agent.get(), env.get(), numEpisodes, &rewardStat);

	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
	//rltl::impl::train(env, agent, numEpisodes, &rewardStat);
	rltl::impl::NN_saveModule(actionValueNet->module(), filename);


}

//void test_deep_sarsa()
//{
//	//rltl::impl::Callback* stepCallback = new TestStepCallback;
//	typedef CartPole Env;
//	Env env;
//
//	rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
//	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
//	auto greedyAction = rltl::impl::MakeGreedyAction(actionValueFunction);
//	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(greedyAction, 0.1f);
//	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 0.2, 0.01, 500 * 200);
//	rltl::impl::DeepSarsaOptions options(0.98, 500, 64, 5, 5);
//	options.multiStep(4);// .multiStepCompound(true);
//
//	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
//	auto agent = rltl::impl::MakeDeepSarsa(replayMemory, actionValueFunction, optimizer, epsilonGreedy, options);
//	RewardStat2 rewardStat;
//	rltl::impl::LinearDecayEpsilonGreedyCallback<decltype(epsilonGreedy)>decayEpsilon(epsilonGreedy, 0.2, 0.01, 500 * 500);
//	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
//	rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
//	rltl::impl::train(env, agent, 1000, &callback);
//}
//
//void test_deep_expected_sarsa()
//{
//	//rltl::impl::Callback* stepCallback = new TestStepCallback;
//	typedef CartPole Env;
//	Env env;
//
//	rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t> actionValueFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1, false);
//	torch::optim::Adam optimizer(actionValueFunction->parameters(), torch::optim::AdamOptions(1e-3));
//	auto greedyAction = rltl::impl::MakeGreedyAction(actionValueFunction);
//	auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(greedyAction, 0.1f);
//	//auto epsilonGreedy = rltl::impl::MakeLinearDecayEpsilonGreedy(actionValueFunction, 1.0, 0.05, 500 * 200);
//	rltl::impl::DeepExpectedSarsaOptions options(0.98, 500, 64, 5, 5);
//	options.multiStep(4).multiStepCompound(true);
//
//	rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
//	auto agent = rltl::impl::MakeDeepExpectedSarsa(replayMemory, actionValueFunction, optimizer, epsilonGreedy, options);
//	RewardStat2 rewardStat;
//	//rltl::impl::DecayEpsilonCallBack<decltype(epsilonGreedy)> decayEpsilon(epsilonGreedy, 1.0f, 0.1f, 100, 0);
//	//rltl::impl::CompositeCallBack<RewardStat2, decltype(decayEpsilon)> callback(rewardStat, decayEpsilon);
//	rltl::impl::train(env, agent, 1000, &rewardStat);
//}
//
//void test_reinforce()
//{
//	//rltl::impl::Callback* stepCallback = new TestStepCallback;
//	typedef CartPole Env;
//	Env env;
//	rltl::impl::MLPPolicyNet<Env::State_t, Env::Action_t> policyFunction(env.stateSpace().dimension(), env.actionSpace().count(), 128, 1);
//	torch::optim::Adam optimizer(policyFunction->parameters(), torch::optim::AdamOptions(1e-3));
//	rltl::impl::ReinforceOptions options(0.98);
//
//	auto agent = rltl::impl::MakeReinforce(policyFunction, optimizer, policyFunction, options);
//	RewardStat2 rewardStat;
//	rltl::impl::train(env, agent, 1000, &rewardStat);
//}
//
void test_actor_critic()
{
	typedef CartPole Env;
	//typedef MountainCar Env;
	auto env = paf::SharedPtr<Env>::Make();
	//Env env;
	Env::ConcreteStateSpacePtr stateSpacePtr = env->stateSpace();
	Env::ConcreteActionSpacePtr actionSpacePtr = env->actionSpace();

	//auto actionValueFunction = rltl::impl::MLPActionValueNet<Env::State_t, Env::Action_t>::Make(rltl::impl::Vector_dimension(stateSpacePtr->low()), actionSpacePtr->count(), 128, 1, false);

	//torch::optim::Adam optimizer((*actionValueFunction)->parameters(), torch::optim::AdamOptions(1e-3));

	//auto greedyAction = rltl::impl::GreedyAction<Env::State_t, Env::Action_t>::Make(actionValueFunction);
	//auto epsilonGreedy = rltl::impl::EpsilonGreedy<Env::State_t, Env::Action_t>::Make(greedyAction, 0.1f);

	typedef rltl::impl::MLPPolicyNet<Env::State_t, Env::Action_t> ActorNet;
	typedef rltl::impl::MLPStateValueNet<Env::State_t> CriticNet;

	auto actorNet = rltl::impl::MLPPolicyNet<Env::State_t, Env::Action_t>::Make(rltl::impl::Vector_dimension(stateSpacePtr->low()), actionSpacePtr->count(), 128, 1);
	auto criticNet = rltl::impl::MLPStateValueNet<Env::State_t>::Make(rltl::impl::Vector_dimension(stateSpacePtr->low()), 128, 1);

	std::shared_ptr<torch::optim::Adam> optimizer(new torch::optim::Adam({
		torch::optim::OptimizerParamGroup((*actorNet)->parameters(), std::make_unique<torch::optim::AdamOptions>(torch::optim::AdamOptions(1e-3))),
		torch::optim::OptimizerParamGroup((*criticNet)->parameters(), std::make_unique<torch::optim::AdamOptions>(torch::optim::AdamOptions(1e-3)))
		}));

	rltl::impl::DeepActorCriticOptions options(0.98, 16);
	//options.targetNetwork(5);
	//options.experienceReplay(1000, 200, 5);
	//options.prioritizedExperienceReplay(10000, 500, 5, 0.2, 0.5);
	//options.multiStep(4, true);

	//auto epsilonGreedy = rltl::impl::MakeEpsilonGreedy(actorNet, 0.1f);

	//rltl::impl::ReplayMemory<Env::State_t, Env::Action_t> replayMemory(rltl::impl::ReplayMemoryOptions(10000, 1.0, 0.5));
	auto agent = rltl::impl::DeepActorCritic<ActorNet, CriticNet>::Make(actorNet, criticNet, optimizer, actorNet, options);
	uint32_t numEpisodes = 4000;
	RewardStat2 rewardStat(numEpisodes);
	//rltl::impl::train(env, agent, 4000, &rewardStat);
	rltl::impl::Trainer<Env::State_t, Env::Action_t>::TrainEpisodes(agent.get(), env.get(), numEpisodes, &rewardStat);

}

int main()
{
	//test_dqn();
	//test_reinforce();
	//test_table();
	try
	{
		test_dqn("./bb.pth");
		//test_actor_critic();
	}
	catch (const std::exception& e)
	{
		std::cout << e.what();
	}
	//test_dqn("./bb.pth");

	//test_deep_expected_sarsa();
}

//inline static DeepQNetwork<ActionValueNet_t, Policy_t> MakeDeepQNetwork(ActionValueNet_t& valueNet, Optimizer_t& optimizer, Policy_t& policy, const DeepQNetworkOptions& options)
