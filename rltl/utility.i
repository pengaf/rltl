#{
#include "impl/utility.h"
#include "impl/num_array.h"
#}

#import "../../paf/pafcore/Utility.i"

namespace rltl
{
	class NumArray
	{};

	class Space : paf::Introspectable
	{};

	class IndexSpace : Space
	{};

	class VectorSpace : Space
	{};

	enum class EnvironmentStatus
	{
		es_normal,
		es_terminated,
		es_truncated
	};

	class Environment : paf::Introspectable
	{
		override abstract Space^ stateSpace();
		override abstract Space^ actionSpace();
		override abstract NumArray reset(int seed);
		override abstract EnvironmentStatus, float, NumArray step(NumArray const & action);
		override abstract void close();
	};

	class Callback : paf::Introspectable
	{
		override virtual void beginTrain();
		override virtual void beginEpisode(uint32_t episode);
		override virtual void beginStep(uint32_t episode, uint32_t step);
		override virtual void endStep(uint32_t episode, uint32_t step, float reward);
		override virtual void endEpisode(uint32_t episode, uint32_t totalStepsInEpisode, float totalRewardsInEpisode);
		override virtual void endTrain();
	};


}