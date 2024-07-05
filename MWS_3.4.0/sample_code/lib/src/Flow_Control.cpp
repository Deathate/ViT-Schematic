// 日期      版本       維護人員    修改原因.
// 20141006  v01.00.00  吳青華      新程式開發.
// 20150623  v01.01.00  吳青華      新增 reset_parameter() 函式.

#include <iostream>
#include <stdint.h>
#include <sys/time.h>
#include <sys/types.h>
#include "../inc/Flow_Control.h"

uint64_t get_current_time_microsecond();

// 功能: Flow_Control 建構式(自訂參數).
// 回傳值為 void.
// 參數 highest_flow_rate_byte_per_second: 最大流量(byte/second).
// 參數 time_slice_microsecond: 測試流量的時間單位(microsecond).
// 參數 accepted_multiple_value_of_jitter: 可容忍的突波倍數(在 "測試流量的時間單位" 的單位時間內可接受的流量倍數).
Flow_Control::Flow_Control(const uint64_t highest_flow_rate_byte_per_second,
                           const uint64_t time_slice_microsecond,
                           const uint64_t accepted_multiple_value_of_jitter)
{
  uint64_t current_time = get_current_time_microsecond();

  // "最大流量" 不能超過 MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND.
  if (highest_flow_rate_byte_per_second <= MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND)
  {
    this->m_highest_flow_rate_byte_per_second = highest_flow_rate_byte_per_second;
  }
  else
  {
    this->m_highest_flow_rate_byte_per_second = MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND;
  }
  // "測試流量的時間單位" 不能超過 MAX_TIME_SLICE.
  if (time_slice_microsecond <= MAX_TIME_SLICE)
  {
    this->m_time_slice_microsecond = time_slice_microsecond;
  }
  else
  {
    this->m_time_slice_microsecond = MAX_TIME_SLICE;
  }
  // "可容忍的突波倍數" 不能超過 MAX_MULTIPLE_VALUE_OF_JITTER.
  if (accepted_multiple_value_of_jitter <= MAX_MULTIPLE_VALUE_OF_JITTER)
  {
    this->m_accepted_multiple_value_of_jitter = accepted_multiple_value_of_jitter;
  }
  else
  {
    this->m_accepted_multiple_value_of_jitter = MAX_MULTIPLE_VALUE_OF_JITTER;
  }

  // 應為 this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond =
  //        this->m_highest_flow_rate_byte_per_second * 1000000(byte 轉成"百萬分之一byte") / 1000000(second 轉成 microsecond)
  // 乘 1000000 和除 1000000 可以抵銷, 所以簡化成下行程式.
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond = this->m_highest_flow_rate_byte_per_second;
  // "測試流量的時間單位" 內控制的流量 = "百萬分之一 byte/microsecond" * "測試流量的時間單位(microsecond)" * (1 + 可容忍的突波倍數).
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice =
    this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond *
    this->m_time_slice_microsecond *
    (1 + this->m_accepted_multiple_value_of_jitter);

  this->m_time_slice_start_time_microsecond = current_time;
  this->m_time_slice_accumulated_flow_byte = 0;

  this->m_second_start_time_microsecond = current_time;
  this->m_second_accumulated_flow_byte = 0;

  this->m_total_start_time_microsecond = current_time;
  this->m_total_accumulated_flow_byte = 0;

  return;
}

// 功能: Flow_Control 建構式(預設參數).
// 回傳值為 void.
// 沒有參數.
Flow_Control::Flow_Control()
{
  uint64_t current_time = get_current_time_microsecond();

  this->m_highest_flow_rate_byte_per_second = DEFAULT_HIGHEST_FLOW_RATE_BYTE_PER_SECOND;
  this->m_time_slice_microsecond = DEFAULT_TIME_SLICE_MICROSECOND;
  this->m_accepted_multiple_value_of_jitter = DEFAULT_ACCEPTED_MULTIPLE_VALUE_OF_JITTER;

  // 應為 this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond =
  //        this->m_highest_flow_rate_byte_per_second * 1000000(byte 轉成"百萬分之一byte") / 1000000(second 轉成 microsecond)
  // 乘 1000000 和除 1000000 可以抵銷, 所以簡化成下行程式.
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond = this->m_highest_flow_rate_byte_per_second;
  // "測試流量的時間單位" 內控制的流量 = "百萬分之一 byte/microsecond" * "測試流量的時間單位(microsecond)" * 可容忍的突波倍數.
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice =
    this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond *
    this->m_time_slice_microsecond *
    (1 + this->m_accepted_multiple_value_of_jitter);

  this->m_time_slice_start_time_microsecond = current_time;
  this->m_time_slice_accumulated_flow_byte = 0;

  this->m_second_start_time_microsecond = current_time;
  this->m_second_accumulated_flow_byte = 0;

  this->m_total_start_time_microsecond = current_time;
  this->m_total_accumulated_flow_byte = 0;

  return;
}

// 功能: Flow_Control 建構式(複製).
// 回傳值為 void.
// 參數 object: 要被複製 Flow_Control 物件.
Flow_Control::Flow_Control(const Flow_Control& object)
{
  this->m_highest_flow_rate_byte_per_second = object.m_highest_flow_rate_byte_per_second;
  this->m_time_slice_microsecond = object.m_time_slice_microsecond;
  this->m_accepted_multiple_value_of_jitter = object.m_accepted_multiple_value_of_jitter;

  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond =
    object.m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond;
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice =
    object.m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice;

  this->m_time_slice_start_time_microsecond = object.m_time_slice_start_time_microsecond;
  this->m_time_slice_accumulated_flow_byte = object.m_time_slice_accumulated_flow_byte;

  this->m_second_start_time_microsecond = object.m_second_start_time_microsecond;
  this->m_second_accumulated_flow_byte = object.m_second_accumulated_flow_byte;

  this->m_total_start_time_microsecond = object.m_total_start_time_microsecond;
  this->m_total_accumulated_flow_byte = object.m_total_accumulated_flow_byte;

  return;
}

// 功能: Flow_Control 解構式.
// 回傳值為 void.
// 沒有參數.
Flow_Control::~Flow_Control()
{
  return;
}

// 功能: 查詢適用的最大的 message size(byte).
// 回傳值: 適用的最大 message size(byte).
// 沒有參數.
// 說明: 適用的最大的 message size = (測試流量的時間單位) * ("測試流量的時間單位"內控制的流量).
uint64_t Flow_Control::show_max_fitting_message_size()
{
  return (this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice / 1000000);
}

// 功能: 查詢 "測試流量的時間單位" 內有多少容量可以傳送資料.
// 回傳值: 有多少 byte 容量可以傳送資料.
// 沒有參數.
uint64_t Flow_Control::show_time_slice_remaining_capacity()
{
  this->check_and_adjust_start_time();

  // 如果這一秒的容量已經用完, 則回傳 0.
  if (this->m_highest_flow_rate_byte_per_second <=
      this->m_second_accumulated_flow_byte)
  {
    return 0;
  }

  // 如果這一個 time slice 的容量已經用完, 則回傳 0.
  if ((this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice / 1000000) <=
      this->m_time_slice_accumulated_flow_byte)
  {
    return 0;
  }

  return ((this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice / 1000000) -
          this->m_time_slice_accumulated_flow_byte);
}

// 功能: 查詢 "一秒" 內有多少容量可以傳送資料.
// 回傳值: 有多少 byte 容量可以傳送資料.
// 沒有參數.
uint64_t Flow_Control::show_second_remaining_capacity()
{
  this->check_and_adjust_start_time();

  // 如果這一秒的容量已經用完, 則回傳 0.
  if (this->m_highest_flow_rate_byte_per_second <=
      this->m_second_accumulated_flow_byte)
  {
    return 0;
  }

  return (this->m_highest_flow_rate_byte_per_second -
          this->m_second_accumulated_flow_byte);
}

// 功能: 使用剩餘流量.
// 回傳值為 void.
// 參數 message_size: 要傳送的 message size.
// 說明: 將 message size 填入本函式表示強迫使用流量(不論剩餘流量是否足夠).
void Flow_Control::consume_remaining_capacity(uint64_t message_size)
{
  this->check_and_adjust_start_time();

  this->m_time_slice_accumulated_flow_byte += message_size;
  this->m_second_accumulated_flow_byte += message_size;
  this->m_total_accumulated_flow_byte += message_size;

  return;
}

// 功能: 測試是否有足夠的剩餘流量傳送資料.
// 回傳值 true: 剩餘流量足夠.
// 回傳值 false: 剩餘流量不足.
// 參數 message_size: 要傳送的 message size.
bool Flow_Control::flow_rate_test(const uint64_t message_size)
{
  this->check_and_adjust_start_time();

  // 如果這一秒的容量不夠, 則回傳 false.
  if (this->m_highest_flow_rate_byte_per_second <=
      (this->m_second_accumulated_flow_byte + message_size))
  {
    return false;
  }

  // 如果這一個 time slice 的容量已經用完, 則回傳 false.
  if ((this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice / 1000000) <=
      (this->m_time_slice_accumulated_flow_byte + message_size))
  {
    return false;
  }

  return true;
}

// 功能: 測試是否有足夠的剩餘流量傳送資料.
// 回傳值 true: 剩餘流量足夠.
// 回傳值 false: 剩餘流量不足.
// 參數 message_size: 要傳送的 message size.
// 參數 required_delay_time: 如果現在沒有足夠剩餘流量,
//                           到下次有剩餘流量要等待的時間(microsecond).
bool Flow_Control::flow_rate_test(const uint64_t message_size,
                                  useconds_t& required_delay_time)
{
  useconds_t gap_to_next_second = 0;
  useconds_t gap_to_next_slice = 0;
  this->check_and_adjust_start_time(gap_to_next_second, gap_to_next_slice);

  // 如果這一秒的容量不夠, 則回傳 false.
  if (this->m_highest_flow_rate_byte_per_second <=
      (this->m_second_accumulated_flow_byte + message_size))
  {
    required_delay_time = gap_to_next_second;
    return false;
  }

  // 如果這一個 time slice 的容量已經用完, 則回傳 false.
  if ((this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice / 1000000) <=
      (this->m_time_slice_accumulated_flow_byte + message_size))
  {
    required_delay_time = gap_to_next_slice;
    return false;
  }

  return true;
}

// 功能: 測試是否有足夠的剩餘流量傳送資料,
//       如果足夠則使用剩餘流量,
//       如果不足夠則不使用剩餘流量.
// 回傳值 true: 剩餘流量足夠, 使用剩餘流量.
// 回傳值 false: 剩餘流量不足, 不使用剩餘流量.
// 參數 message_size: 要傳送的 message size.
bool Flow_Control::flow_rate_test_and_consume_remaining_capacity(const uint64_t message_size)
{
  // 如果剩餘流量不足夠則不使用剩餘流量, 回傳 false.
  if (this->flow_rate_test(message_size) == false)
  {
    return false;
  }

  // 剩餘流量足夠則使用剩餘流量, 回傳 true.
  this->m_time_slice_accumulated_flow_byte += message_size;
  this->m_second_accumulated_flow_byte += message_size;
  this->m_total_accumulated_flow_byte += message_size;

  return true;
}

// 功能: 測試是否有足夠的剩餘流量傳送資料,
//       如果足夠則使用剩餘流量,
//       如果不足夠則不使用剩餘流量.
// 回傳值 true: 剩餘流量足夠, 使用剩餘流量.
// 回傳值 false: 剩餘流量不足, 不使用剩餘流量.
// 參數 message_size: 要傳送的 message size.
// 參數 required_delay_time: 如果現在沒有足夠剩餘流量,
//                           到下次有剩餘流量要等待的時間(microsecond).
bool Flow_Control::flow_rate_test_and_consume_remaining_capacity(const uint64_t message_size,
                                                                 useconds_t& required_delay_time)
{
  // 如果剩餘流量不足夠則不使用剩餘流量, 回傳 false.
  if (this->flow_rate_test(message_size, required_delay_time) == false)
  {
    return false;
  }

  // 剩餘流量足夠則使用剩餘流量, 回傳 true.
  this->m_time_slice_accumulated_flow_byte += message_size;
  this->m_second_accumulated_flow_byte += message_size;
  this->m_total_accumulated_flow_byte += message_size;

  return true;
}

// 功能: 查詢最大流量(byte/second).
// 回傳值: 最大流量(byte/second).
// 沒有參數.
uint64_t Flow_Control::show_highest_flow_rate_byte_per_second()
{
  return this->m_highest_flow_rate_byte_per_second;
}

// 功能: 查詢 "測試流量的時間單位(microsecond)".
// 回傳值: 測試流量的時間單位(microsecond).
// 沒有參數.
uint64_t Flow_Control::show_time_slice_microsecond()
{
  return this->m_time_slice_microsecond;
}

// 功能: 查詢 "可容忍的突波倍數".
// 回傳值: 可容忍的突波倍數.
// 沒有參數.
uint64_t Flow_Control::show_accepted_multiple_value_of_jitter()
{
  return this->m_accepted_multiple_value_of_jitter;
}

// 功能: 查詢 "總累計流量起算時間(microsecond)".
// 回傳值: 總累計流量起算時間(microsecond).
// 沒有參數.
uint64_t Flow_Control::show_total_start_time_microsecond()
{
  return this->m_total_start_time_microsecond;
}

// 功能: 查詢 "總累計流量(byte)".
// 回傳值: 總累計流量(byte).
// 沒有參數.
uint64_t Flow_Control::show_total_accumulated_flow_byte()
{
  return this->m_total_accumulated_flow_byte;
}

// 功能: 查詢 "以 microsecond 為單位" 內控制的流量(one millionth part of byte/microsecond)".
// 回傳值: 以 microsecond 為單位" 內控制的流量(one millionth part of byte/microsecond).
// 沒有參數.
uint64_t Flow_Control::show_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond()
{
  return this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond;
}

// 功能: 查詢 "測試流量的時間單位" 內控制的流量(one millionth part of byte/time slice)".
// 回傳值: 測試流量的時間單位" 內控制的流量(one millionth part of byte/time slice).
// 沒有參數.
uint64_t Flow_Control::show_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice()
{
  return this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice;
}

// 功能: 查詢 "測試流量的時間單位起算時間(microsecond)".
// 回傳值: 測試流量的時間單位起算時間(microsecond).
// 沒有參數.
uint64_t Flow_Control::show_time_slice_start_time_microsecond()
{
  return this->m_time_slice_start_time_microsecond;
}

// 功能: 查詢 "測試流量的時間單位的單位時間內的累計流量(byte)".
// 回傳值: 測試流量的時間單位的單位時間內的累計流量(byte).
// 沒有參數.
uint64_t Flow_Control::show_time_slice_accumulated_flow_byte()
{
  return this->m_time_slice_accumulated_flow_byte;
}

// 功能: 查詢 "以秒為單位的累計流量起算時間(microsecond)".
// 回傳值: 以秒為單位的累計流量起算時間(microsecond).
// 沒有參數.
uint64_t Flow_Control::show_second_start_time_microsecond()
{
  return this->m_second_start_time_microsecond;
}

// 功能: 查詢 "以秒為單位的累計流量(byte)".
// 回傳值: 以秒為單位的累計流量(byte).
// 沒有參數.
uint64_t Flow_Control::show_second_accumulated_flow_byte()
{
  return this->m_second_accumulated_flow_byte;
}

// 功能: 檢核及調整 "測試流量的時間單位起算時間" 及 "以秒為單位的累計流量起算時間".
// 回傳值為 void.
// 沒有參數.
void Flow_Control::check_and_adjust_start_time()
{
  uint64_t current_time = get_current_time_microsecond();

  if (current_time >= (this->m_second_start_time_microsecond + 1000000))
  {
    this->m_time_slice_start_time_microsecond = current_time;
    this->m_time_slice_accumulated_flow_byte = 0;

    this->m_second_start_time_microsecond = current_time;
    this->m_second_accumulated_flow_byte = 0;
  }
  else if (current_time >= (this->m_time_slice_start_time_microsecond + this->m_time_slice_microsecond))
  {
    this->m_time_slice_start_time_microsecond = current_time;
    this->m_time_slice_accumulated_flow_byte = 0;
  }

  return;
}

// 功能: 檢核及調整 "測試流量的時間單位起算時間" 及 "以秒為單位的累計流量起算時間".
// 回傳值為 void.
// 參數 gap_to_next_second: 到下一秒所需的時間(microsecond).
// 參數 gap_to_next_slice: 到下一個 slice 所需的時間(microsecond)..
void Flow_Control::check_and_adjust_start_time(useconds_t& gap_to_next_second,
                                               useconds_t& gap_to_next_slice)
{
  uint64_t current_time = get_current_time_microsecond();

  if (current_time >= (this->m_second_start_time_microsecond + 1000000))
  {
    this->m_time_slice_start_time_microsecond = current_time;
    this->m_time_slice_accumulated_flow_byte = 0;

    this->m_second_start_time_microsecond = current_time;
    this->m_second_accumulated_flow_byte = 0;
  }
  else if (current_time >= (this->m_time_slice_start_time_microsecond + this->m_time_slice_microsecond))
  {
    this->m_time_slice_start_time_microsecond = current_time;
    this->m_time_slice_accumulated_flow_byte = 0;
  }

  gap_to_next_second = (this->m_second_start_time_microsecond + 1000000) - current_time;
  gap_to_next_slice = (this->m_time_slice_start_time_microsecond + this->m_time_slice_microsecond) - current_time;

  return;
}

// 功能: 重新設定參數.
// 回傳值為 void.
// 參數 new_highest_flow_rate_byte_per_second: 新的最大流量(byte/second).
// 參數 new_time_slice_microsecond: 新的測試流量的時間單位(microsecond).
// 參數 new_accepted_multiple_value_of_jitter: 新的可容忍的突波倍數(在 "測試流量的時間單位" 的單位時間內可接受的流量倍數).
void Flow_Control::reset_parameter(const uint64_t new_highest_flow_rate_byte_per_second,
                                   const uint64_t new_time_slice_microsecond,
                                   const uint64_t new_accepted_multiple_value_of_jitter)
{
  uint64_t current_time = get_current_time_microsecond();

  // "最大流量" 不能超過 MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND.
  if (new_highest_flow_rate_byte_per_second <= MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND)
  {
    this->m_highest_flow_rate_byte_per_second = new_highest_flow_rate_byte_per_second;
  }
  else
  {
    this->m_highest_flow_rate_byte_per_second = MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND;
  }
  // "測試流量的時間單位" 不能超過 MAX_TIME_SLICE.
  if (new_time_slice_microsecond <= MAX_TIME_SLICE)
  {
    this->m_time_slice_microsecond = new_time_slice_microsecond;
  }
  else
  {
    this->m_time_slice_microsecond = MAX_TIME_SLICE;
  }
  // "可容忍的突波倍數" 不能超過 MAX_MULTIPLE_VALUE_OF_JITTER.
  if (new_accepted_multiple_value_of_jitter <= MAX_MULTIPLE_VALUE_OF_JITTER)
  {
    this->m_accepted_multiple_value_of_jitter = new_accepted_multiple_value_of_jitter;
  }
  else
  {
    this->m_accepted_multiple_value_of_jitter = MAX_MULTIPLE_VALUE_OF_JITTER;
  }

  // 應為 this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond =
  //        this->m_highest_flow_rate_byte_per_second * 1000000(byte 轉成"百萬分之一byte") / 1000000(second 轉成 microsecond)
  // 乘 1000000 和除 1000000 可以抵銷, 所以簡化成下行程式.
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond = this->m_highest_flow_rate_byte_per_second;
  // "測試流量的時間單位" 內控制的流量 = "百萬分之一 byte/microsecond" * "測試流量的時間單位(microsecond)" * (1 + 可容忍的突波倍數).
  this->m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice =
    this->m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond *
    this->m_time_slice_microsecond *
    (1 + this->m_accepted_multiple_value_of_jitter);

  this->m_time_slice_start_time_microsecond = current_time;
  this->m_time_slice_accumulated_flow_byte = 0;

  this->m_second_start_time_microsecond = current_time;
  this->m_second_accumulated_flow_byte = 0;

  this->m_total_start_time_microsecond = current_time;
  this->m_total_accumulated_flow_byte = 0;

  return;
}

uint64_t get_current_time_microsecond()
{
  struct timeval tim;
  gettimeofday(&tim, NULL);

  return (((uint64_t)tim.tv_sec * (uint64_t)1000000) + (uint64_t)tim.tv_usec);
}
