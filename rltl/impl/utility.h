#pragma once
#include <stdint.h>
#include <assert.h>
#include <vector>
#include  "../../../paf/pafcore/SmartPtr.h"
#include <torch/torch.h>

#define BEGIN_RLTL_IMPL	namespace rltl { namespace impl {
#define END_RLTL_IMPL	} }

BEGIN_RLTL_IMPL

typedef torch::Tensor Tensor;
typedef torch::nn::Module Module;
typedef torch::optim::Optimizer Optimizer;

enum class SpaceCategory
{
	index_space,//integer
	vector_space,//1-d tensor
	image_space,//3-d tensor
};

enum class EnvironmentStatus
{
	es_normal,
	es_terminated,
	es_truncated
};

enum class ExperienceReplay
{
	no_experience_replay,
	experience_replay,
	prioritized_experience_replay,
};

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


template<typename Element_t>
class Space : public paf::Introspectable
{
public:
	//typedef Element_t Element_t;
public:
	virtual SpaceCategory category() = 0;
	//virtual Element_t sample() = 0;
};


template<typename Element_t = uint32_t>
class IndexSpace : public Space<Element_t>
{
public:
	IndexSpace(uint32_t count) :
		m_count(count)
	{}
public:
	SpaceCategory category()
	{
		return SpaceCategory::index_space;
	}
	size_t count() const
	{
		return m_count;
	}
public:
	uint32_t m_count;
};

template<typename Element_t>
class VectorSpace : public Space<Element_t>
{
public:
	//typedef Element_t Element_t;
public:
	VectorSpace(Element_t low, Element_t high) :
		m_low(low),
		m_high(high)
	{}
public:
	SpaceCategory category()
	{
		return SpaceCategory::vector_space;
	}
public:
	const Element_t& low() const
	{
		return m_low;
	}
	const Element_t& high() const
	{
		return m_high;
	}
protected:
	Element_t m_low;
	Element_t m_high;
};

template<typename Element_t>
class ImageSpace : public Space<Element_t>
{
public:
	//typedef Element_t Element_t;
public:
	SpaceCategory category()
	{
		return SpaceCategory::image_space;
	}
public:
	float low() const
	{
		return m_low;
	}
	float high() const
	{
		return m_high;
	}
protected:
	float m_low;
	float m_high;
};


template<typename State_t, typename Action_t>
class Environment : public paf::Introspectable
{
public:
	//typedef State_t State_t;
	//typedef Action_t Action_t;
	typedef Space<State_t> StateSpace_t;
	typedef Space<Action_t> ActionSpace_t;
	typedef paf::SharedPtr<StateSpace_t> StateSpacePtr;
	typedef paf::SharedPtr<ActionSpace_t> ActionSpacePtr;
public:
	virtual StateSpacePtr stateSpace() = 0;
	virtual ActionSpacePtr actionSpace() = 0;
	virtual State_t reset(int seed = 0) = 0;
	virtual EnvironmentStatus step(float& reward, State_t& nextState, const Action_t& action) = 0;
	virtual void close() = 0;
};


template<typename State_t, typename Action_t>
class ActionValueFunction : public paf::Introspectable
{
public:
	//typedef State_t State_t;
	//typedef Action_t Action_t;
public:
	virtual Action_t maxAction(const State_t& state, bool firstMax = true) = 0;
	virtual void getValues(std::vector<float>& values, const State_t& state) = 0;
	virtual uint32_t actionCount() const = 0;
};

//template<typename State_t, typename Action_t>
//class DiscreteActionValueFunction : public ActionValueFunction<State_t, Action_t>
//{
//public:
//	virtual uint32_t actionCount() const = 0;
//};

template<typename State_t>
class StateValueFunction : public paf::Introspectable
{
public:
	//typedef State_t State_t;
public:
	virtual float getValue(const State_t& state) = 0;
};

template<typename State_t, typename Action_t>
class PolicyFunction : public paf::Introspectable
{
public:
	//typedef State_t State_t;
	//typedef Action_t Action_t;
public:
	virtual Action_t takeAction(const State_t& state) = 0;
	virtual uint32_t actionCount() const = 0;
};

template<typename State_t, typename Action_t>
class DiscretePolicyFunction : public PolicyFunction<State_t, Action_t>
{
public:
	//virtual uint32_t actionCount() const = 0;
};

template<typename State_t, typename Action_t>
class ActionValueNet : public ActionValueFunction<State_t, Action_t>
{
public:
	virtual Tensor forward(const Tensor& stateTensor) = 0;
	virtual Module* module() = 0;
};

template<typename State_t>
class StateValueNet : public StateValueFunction<State_t>
{
public:
	virtual Tensor forward(const Tensor& stateTensor) = 0;
	virtual Module* module() = 0;
};

template<typename State_t, typename Action_t>
class PolicyNet : public PolicyFunction<State_t, Action_t>
{
public:
	virtual Tensor forward(const Tensor& stateTensor) = 0;
	virtual Module* module() = 0;
};

template<typename State_t, typename Action_t>
class PolicyStateValueNet : public PolicyFunction<State_t, Action_t>, public StateValueFunction<State_t>
{
public:
	virtual void forward(Tensor& actionTensor, Tensor& stateValueTensor, const Tensor& stateTensor) = 0;
};

template<typename State_t, typename Action_t>
class PolicyActionValueNet : public PolicyFunction<State_t, Action_t>, public ActionValueFunction<State_t, Action_t>
{
public:
	virtual void forward(Tensor& actionTensor, Tensor& actionValueTensor, const Tensor& stateTensor) = 0;
};

template<typename State_t, typename Action_t>
class Agent : public paf::Introspectable
{
public:
	//typedef State_t State_t;
	//typedef Action_t Action_t;
public:
	virtual Action_t firstStep(const State_t& firstState) = 0;
	virtual Action_t nextStep(float reward, const State_t& nextState) = 0;
	virtual void lastStep(float reward, const State_t& nextState, bool terminated) = 0;
};

class Callback : public paf::Introspectable
{
public:
	virtual void beginTrain()
	{}
	virtual void beginEpisode(uint32_t episode)
	{}
	virtual void beginStep(uint32_t episode, uint32_t step)
	{}
	virtual void endStep(uint32_t episode, uint32_t step, float reward)
	{}
	virtual void endEpisode(uint32_t episode, uint32_t totalStepsInEpisode, float totalRewardsInEpisode)
	{}
	virtual void endTrain()
	{}
};


END_RLTL_IMPL