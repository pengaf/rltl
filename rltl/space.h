#pragma once
#include <string>
#include <vector>
#include "data_type.h"

namespace rltl
{
    class Space
    {
	public:
		Space(const char* name);
    public:
		const char* get_name() const
		{
			return m_name.c_str();
		}
    protected:
		std::string m_name;
    };

	class Element
	{
	public:
		Space* m_space;
	};

	class DiscreteElement : public Element
	{
		int32_t value;
	};

	class DiscreteSpace : public Space
	{
	public:
		DiscreteSpace(const char* name, uint32_t count, int32_t start = 0);
	public:
		int32_t start() const
		{
			return m_start;
		}
		uint32_t count() const
		{
			return m_count;
		}
	protected:
		int32_t m_start{};
		uint32_t m_count{};
	};

	class NormalizedDiscreteSpace : public Space
	{
	public:
		NormalizedDiscreteSpace(const char* name, uint32_t count);
	public:
		uint32_t count() const
		{
			return m_count;
		}
	public:
		uint32_t m_count;
	};



	class ContinuousSpace : public Space
	{
	public:
		ContinuousSpace(const char* name, float low, float high, DataType dataType = DataType::dt_float32);
	protected:
		float m_low;
		float m_high;
		DataType m_dataType{};
		uint8_t m_elementSize{};
	};

	class MultiDiscreteSpace : public Space
	{
	public:
		MultiDiscreteSpace(const char* name, const std::vector<size_t>& shape, uint64_t count, int64_t start = 0);
	public:
		int64_t element_count() const
		{
			return m_end - m_start;
		}
		int64_t element_size() const
		{
			return m_dataType.data_size();
		}
		DataType data_type() const
		{
			return m_dataType;
		}
	protected:
		int64_t m_start{};
		int64_t m_end{};
		DataType m_dataType;
		std::vector<size_t> m_shape;
	};



	class CompositeSpace : public Space
	{
	public:
		CompositeSpace(const char* name);
	public:
		void addSubSpace(Space* space);
		size_t subSpaceCount() const;
		Space* getSubSpace(size_t index) const;
	protected:
		std::vector<Space*> m_spaces;
	};




	template<typename T>
	class TSpace
	{
	public:
		TSpace(T start, T end);
	public:
		T m_start;
		T m_end;
	};
}