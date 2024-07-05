// 日期      版本       維護人員    修改原因.
// 20141006  v01.00.00  吳青華      新程式開發.
// 20150623  v01.01.00  吳青華      新增 reset_parameter() 函式.

#ifndef FLOW_CONTROL_H_
#define FLOW_CONTROL_H_

#include <iostream>
#include <stdint.h>
#include <sys/types.h>

// 可設定的 "最大流量" 不能超過 10 gigabyte/second(10,000,000,000 byte/second).
#define MAX_HIGHEST_FLOW_RATE_BYTE_PER_SECOND 10000000000
// 可設定的 "測試流量的時間單位" 不能超過 100000 microsecond(0.1 second).
#define MAX_TIME_SLICE 100000
// 可設定的 "可容忍的突波倍數" 不能超過 5 倍.
#define MAX_MULTIPLE_VALUE_OF_JITTER 5
// 預設的 "最大流量" 為 1 megabyte/second(1,000,000 byte/second).
#define DEFAULT_HIGHEST_FLOW_RATE_BYTE_PER_SECOND 1000000
// 預設的 "測試流量的時間單位" 為 10 millisecond.
#define DEFAULT_TIME_SLICE_MICROSECOND 10000
// 預設的 "可容忍的突波倍數" 為 1 倍.
#define DEFAULT_ACCEPTED_MULTIPLE_VALUE_OF_JITTER 1

class Flow_Control
{
  public:
    // 功能: Flow_Control 建構式(自訂參數).
    // 回傳值為 void.
    // 參數 highest_flow_rate_byte_per_second: 最大流量(byte/second).
    // 參數 time_slice_microsecond: 測試流量的時間單位(microsecond).
    // 參數 accepted_multiple_value_of_jitter: 可容忍的突波倍數(在 "測試流量的時間單位" 的單位時間內可接受的流量倍數).
    Flow_Control(const uint64_t highest_flow_rate_byte_per_second,
                 const uint64_t time_slice_microsecond,
                 const uint64_t accepted_multiple_value_of_jitter);
    // 功能: Flow_Control 建構式(預設參數).
    // 回傳值為 void.
    // 沒有參數.
    Flow_Control();
    // 功能: Flow_Control 建構式(複製).
    // 回傳值為 void.
    // 參數 object: 要被複製 Flow_Control 物件.
    Flow_Control(const Flow_Control& object);
    // 功能: Flow_Control 解構式.
    // 回傳值為 void.
    // 沒有參數.
    ~Flow_Control();

    // 功能: 查詢適用的最大的 message size(byte).
    // 回傳值: 適用的最大 message size(byte).
    // 沒有參數.
    // 說明: 適用的最大的 message size = (測試流量的時間單位) * ("測試流量的時間單位"內控制的流量).
    uint64_t show_max_fitting_message_size();

    // 功能: 查詢 "測試流量的時間單位" 內有多少容量可以傳送資料.
    // 回傳值: 有多少 byte 容量可以傳送資料.
    // 沒有參數.
    uint64_t show_time_slice_remaining_capacity();
    // 功能: 查詢 "一秒" 內有多少容量可以傳送資料.
    // 回傳值: 有多少 byte 容量可以傳送資料.
    // 沒有參數.
    uint64_t show_second_remaining_capacity();
    // 功能: 使用剩餘流量.
    // 回傳值為 void.
    // 參數 message_size: 要傳送的 message size.
    // 說明: 將 message size 填入本函式表示強迫使用流量(不論剩餘流量是否足夠).
    void consume_remaining_capacity(uint64_t message_size);
    // 功能: 測試是否有足夠的剩餘流量傳送資料.
    // 回傳值 true: 剩餘流量足夠.
    // 回傳值 false: 剩餘流量不足.
    // 參數 message_size: 要傳送的 message size.
    bool flow_rate_test(const uint64_t message_size);
    // 功能: 測試是否有足夠的剩餘流量傳送資料.
    // 回傳值 true: 剩餘流量足夠.
    // 回傳值 false: 剩餘流量不足.
    // 參數 message_size: 要傳送的 message size.
    // 參數 required_delay_time: 如果現在沒有足夠剩餘流量,
    //                           到下次有剩餘流量要等待的時間(microsecond).
    bool flow_rate_test(const uint64_t message_size,
                        useconds_t& required_delay_time);
    // 功能: 測試是否有足夠的剩餘流量傳送資料,
    //       如果足夠則使用剩餘流量,
    //       如果不足夠則不使用剩餘流量.
    // 回傳值 true: 剩餘流量足夠, 使用剩餘流量.
    // 回傳值 false: 剩餘流量不足, 不使用剩餘流量.
    // 參數 message_size: 要傳送的 message size.
    bool flow_rate_test_and_consume_remaining_capacity(const uint64_t message_size);
    // 功能: 測試是否有足夠的剩餘流量傳送資料,
    //       如果足夠則使用剩餘流量,
    //       如果不足夠則不使用剩餘流量.
    // 回傳值 true: 剩餘流量足夠, 使用剩餘流量.
    // 回傳值 false: 剩餘流量不足, 不使用剩餘流量.
    // 參數 message_size: 要傳送的 message size.
    // 參數 required_delay_time: 如果現在沒有足夠剩餘流量,
    //                           到下次有剩餘流量要等待的時間(microsecond).
    bool flow_rate_test_and_consume_remaining_capacity(const uint64_t message_size,
                                                       useconds_t& required_delay_time);

    // 功能: 查詢最大流量(byte/second).
    // 回傳值: 最大流量(byte/second).
    // 沒有參數.
    uint64_t show_highest_flow_rate_byte_per_second();
    // 功能: 查詢 "測試流量的時間單位(microsecond)".
    // 回傳值: 測試流量的時間單位(microsecond).
    // 沒有參數.
    uint64_t show_time_slice_microsecond();
    // 功能: 查詢 "可容忍的突波倍數".
    // 回傳值: 可容忍的突波倍數.
    // 沒有參數.
    uint64_t show_accepted_multiple_value_of_jitter();
    // 功能: 查詢 "總累計流量起算時間(microsecond)".
    // 回傳值: 總累計流量起算時間(microsecond).
    // 沒有參數.
    uint64_t show_total_start_time_microsecond();
    // 功能: 查詢 "總累計流量(byte)".
    // 回傳值: 總累計流量(byte).
    // 沒有參數.
    uint64_t show_total_accumulated_flow_byte();

    // 功能: 查詢 "以 microsecond 為單位" 內控制的流量(one millionth part of byte/microsecond)".
    // 回傳值: 以 microsecond 為單位" 內控制的流量(one millionth part of byte/microsecond).
    // 沒有參數.
    uint64_t show_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond();
    // 功能: 查詢 "測試流量的時間單位" 內控制的流量(one millionth part of byte/time slice)".
    // 回傳值: 測試流量的時間單位" 內控制的流量(one millionth part of byte/time slice).
    // 沒有參數.
    uint64_t show_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice();

    // 功能: 查詢 "測試流量的時間單位起算時間(microsecond)".
    // 回傳值: 測試流量的時間單位起算時間(microsecond).
    // 沒有參數.
    uint64_t show_time_slice_start_time_microsecond();
    // 功能: 查詢 "測試流量的時間單位的單位時間內的累計流量(byte)".
    // 回傳值: 測試流量的時間單位的單位時間內的累計流量(byte).
    // 沒有參數.
    uint64_t show_time_slice_accumulated_flow_byte();
    // 功能: 查詢 "以秒為單位的累計流量起算時間(microsecond)".
    // 回傳值: 以秒為單位的累計流量起算時間(microsecond).
    // 沒有參數.
    uint64_t show_second_start_time_microsecond();
    // 功能: 查詢 "以秒為單位的累計流量(byte)".
    // 回傳值: 以秒為單位的累計流量(byte).
    // 沒有參數.
    uint64_t show_second_accumulated_flow_byte();

    // 功能: 重新設定參數.
    // 回傳值為 void.
    // 參數 new_highest_flow_rate_byte_per_second: 新的最大流量(byte/second).
    // 參數 new_time_slice_microsecond: 新的測試流量的時間單位(microsecond).
    // 參數 new_accepted_multiple_value_of_jitter: 新的可容忍的突波倍數(在 "測試流量的時間單位" 的單位時間內可接受的流量倍數).
    void reset_parameter(const uint64_t new_highest_flow_rate_byte_per_second,
                         const uint64_t new_time_slice_microsecond,
                         const uint64_t new_accepted_multiple_value_of_jitter);

  private:
    // 參數: 最大流量(byte/second).
    uint64_t m_highest_flow_rate_byte_per_second;
    // 參數: 測試流量的時間單位(microsecond).
    uint64_t m_time_slice_microsecond;
    // 參數: 可容忍的突波倍數(在 "測試流量的時間單位" 的單位時間內可接受的流量倍數).
    uint64_t m_accepted_multiple_value_of_jitter;

    // "以 microsecond 為單位" 內控制的流量(one millionth part of byte/microsecond).
    uint64_t m_controlled_flow_rate_one_millionth_part_of_byte_per_microsecond;
    // "測試流量的時間單位" 內控制的流量(one millionth part of byte/time slice).
    uint64_t m_controlled_flow_rate_one_millionth_part_of_byte_per_time_slice;

    // 測試流量的時間單位起算時間(microsecond).
    uint64_t m_time_slice_start_time_microsecond;
    // 測試流量的時間單位的單位時間內的累計流量(byte).
    uint64_t m_time_slice_accumulated_flow_byte;

    // 以秒為單位的累計流量起算時間(microsecond).
    uint64_t m_second_start_time_microsecond;
    // 以秒為單位的累計流量(byte).
    uint64_t m_second_accumulated_flow_byte;

    // 總累計流量起算時間(microsecond).
    uint64_t m_total_start_time_microsecond;
    // 總累計流量(byte).
    uint64_t m_total_accumulated_flow_byte;

    // 功能: 檢核及調整 "測試流量的時間單位起算時間" 及 "以秒為單位的累計流量起算時間".
    // 回傳值為 void.
    // 沒有參數.
    void check_and_adjust_start_time();

    // 功能: 檢核及調整 "測試流量的時間單位起算時間" 及 "以秒為單位的累計流量起算時間".
    // 回傳值為 void.
    // 參數 gap_to_next_second: 到下一秒所需的時間(microsecond).
    // 參數 gap_to_next_slice: 到下一個 slice 所需的時間(microsecond)..
    void check_and_adjust_start_time(useconds_t& gap_to_next_second,
                                     useconds_t& gap_to_next_slice);
};

#endif // FLOW_CONTROL_H_
