
namespace rltl
{
	template<State_t, Value_t>
	class StateValueFunction
	{
#{
	public:
		typedef State_t State_t;
		typedef Value_t Value_t;
#}
		virtual Value_t getValue(State_t const& state) const;
		virtual void setValue(State_t  const& state, Value_t const& value);
	};


	template<State_t, Action_t, Value_t>
	class ActionValueFunction
	{
#{
	public:
		typedef State_t State_t;
		typedef Action_t Action_t;
		typedef Value_t Value_t;
	public:
#}
		virtual Value_t getValue(State_t const& state, Action_t const& action) const;
	};

	template<State_t, Action_t, Value_t>
	class ActionValueTable : ActionValueFunction<State_t, Action_t, Value_t>
	{
	};

}
