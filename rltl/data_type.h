#pragma once
#include <utility>

namespace rltl
{
	class DataType
	{
	public:
		enum
		{
			dt_int8,
			dt_int16,
			dt_int32,
			dt_int64,
			dt_float32,
			dt_float64,
			dt_unknown,
		};

		DataType() : m_dataType(dt_unknown)
		{}

		DataType(int8_t dataType) :
			m_dataType(m_dataType)
		{
			calc_data_size();
		}

		bool operator == (int8_t dataType) const
		{
			return m_dataType == dataType;
		}

		bool operator != (int8_t dataType) const
		{
			return m_dataType != dataType;
		}

		DataType& operator=(int8_t dataType)
		{
			m_dataType = dataType;
			calc_data_size();
			return *this;
		}

		bool is_int() const
		{
			return (dt_int8 == m_dataType || dt_int16 == m_dataType || dt_int32 == m_dataType || dt_int64 == m_dataType);
		}

		bool is_float() const
		{
			return (dt_float32 == m_dataType || dt_float64 == m_dataType);
		}

		size_t data_size() const
		{
			return m_dataSize;
		}
	private:
		void calc_data_size()
		{
			switch (m_dataType)
			{
			case dt_int8:
				m_dataSize = sizeof(int8_t);
				break;
			case dt_int16:
				m_dataSize = sizeof(int16_t);
				break;
			case dt_int32:
				m_dataSize = sizeof(int32_t);
				break;
			case dt_int64:
				m_dataSize = sizeof(int64_t);
				break;
			case dt_float32:
				m_dataSize = sizeof(float);
				break;
			case dt_float64:
				m_dataSize = sizeof(double);
				break;
			default:
				m_dataSize = 0;
			}
		}
	private:
		int8_t m_dataType;
		int8_t m_dataSize;
	};



	//template<DataType>
	//struct DataType_t
	//{};

	//struct DataType_t<dt_int8>
	//{
	//	typedef int8_t data_t;
	//};
	//struct DataType_t<dt_int16>
	//{
	//	typedef int16_t data_t;
	//};
	//struct DataType_t<dt_int32>
	//{
	//	typedef int32_t data_t;
	//};
}