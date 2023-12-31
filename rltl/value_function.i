
namespace rltl
{
	template<State_t, float>
	class StateValueFunction
	{
#{
	public:
		typedef State_t State_t;
		
#}
		virtual float getValue(State_t const& state) const;
		virtual void setValue(State_t  const& state, float const& value);
	};


	template<State_t, Action_t, float>
	class ActionValueFunction
	{
#{
	public:
		typedef State_t State_t;
		typedef Action_t Action_t;
		
	public:
#}
		virtual float getValue(State_t const& state, Action_t const& action) const;
	};

	template<State_t, Action_t, float>
	class ActionValueTable : ActionValueFunction<State_t, Action_t, float>
	{
	};

}
