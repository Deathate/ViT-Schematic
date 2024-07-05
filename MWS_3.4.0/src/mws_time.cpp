// 日期      版本       維護人員    修改原因.
// 20130201  v01.00.00  吳青華      新程式開發.
// 20130508  v01.01.00  吳青華      程式演算方式優化.
// 20150120  v01.02.00  吳青華      新增"判斷是否現在時間是否小於等於參數時間"函式.
// 20170828  v01.03.00  吳青華      NSK, AIX, Linux 三平台共用函式庫修改.

//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_TIME_CPP 1

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

#include <ctime>
#include <sys/time.h>
#include <unistd.h>

#include "../inc/mws_time.h"

mws_time::mws_time()
{

  return;
}

mws_time::~mws_time()
{

  return;
}

// 功能: gmtime() 的 thread safe 版
// 回傳值為 void
// 參數 timer: time_t 時間, 要將此時間轉成 UTC tm 格式時間
// 參數 result: struct tm 格式 local 時區時間.
void mws_time::gmtime_TS(const time_t &timer,
                         struct tm &result)
{
  gmtime_r(&timer, &result);
  return ;
}

// 功能: localtime() 的 thread safe 版
// 回傳值為 void
// 參數 timer: time_t 時間, 要將此時間轉成 local 時區時間
// 參數 result: struct tm 格式 local 時區時間.
void mws_time::localtime_TS(const time_t &timer,
                            struct tm &result)
{
  localtime_r(&timer, &result);
  return ;
}

// 功能: 取得 local 日期
// 回傳值: std::string 型態的 local 日期.
std::string mws_time::get_local_date()
{
  // 先取得標準時間.
  time_t current_time = time(NULL);
  // 將時間轉成 local time
  struct tm ts;
  localtime_TS(current_time, ts);

  std::stringstream ss("00000000");
  ss << std::setw(4) << (ts.tm_year + 1900)
     << std::setw(2) << (ts.tm_mon + 1)
     << std::setw(2) << ts.tm_mday;

  std::string s = ss.str();

  for (size_t i = 0; i < s.length(); ++i)
  {
    if (s[i] == ' ')
    {
      s[i] = '0';
    }
  }

  return s;
}

// 功能: 取得當地現在日期, 格式為 YYYYMMDD
// 回傳值: 格式為 YYYYMMDD 的當地現在日期
// 沒有參數.
unsigned long long int mws_time::get_current_local_date()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return timeval_to_local_date(tim);
}

// 功能: 取得 local 時間
// 回傳值: std::string 型態的 local 時間.
std::string mws_time::get_local_time()
{
  // 先取得標準時間.
  time_t current_time = time(NULL);
  // 將時間轉成 local time
  struct tm ts;
  localtime_TS(current_time, ts);

  std::stringstream ss("00:00:00");
  ss << std::setw(2) << ts.tm_hour
     << std::setw(1) << ":"
     << std::setw(2) << ts.tm_min
     << std::setw(1) << ":"
     << std::setw(2) << ts.tm_sec;

  std::string s = ss.str();

  for (size_t i = 0; i < s.length(); ++i)
  {
    if (s[i] == ' ')
    {
      s[i] = '0';
    }
  }

  return s;
}

// 功能: 取得當地現在時間, 格式為兩位時兩位分兩位秒三位 millisecond 三位 microsecond
// 回傳值: 格式為兩位時兩位分兩位秒三位 millisecond 三位 microsecond 的當地現在時間
// 沒有參數.
unsigned long long int mws_time::get_current_local_time()
{
  timeval tim;
  gettimeofday(&tim, NULL);
  return timeval_to_local_time(tim);
}

// 功能: 取得當地現在時間, 格式為 microsecond
// 回傳值: 格式為 microsecond 的當地現在時間
// 沒有參數.
std::string mws_time::get_local_time_microsecond()
{
  timeval tim;
  gettimeofday(&tim, NULL);

  time_t raw_time = tim.tv_sec;
  struct tm time_info;
  localtime_TS(raw_time, time_info);

  std::stringstream ss("00:00:00.000000");
  ss << std::setw(2) << (unsigned long long int)time_info.tm_hour
     << std::setw(1) << ":"
     << std::setw(2) << (unsigned long long int)time_info.tm_min
     << std::setw(1) << ":"
     << std::setw(2) << (unsigned long long int)time_info.tm_sec
     << std::setw(1) << "."
     << std::setw(6) << tim.tv_usec;

  std::string s = ss.str();

  for (size_t i = 0; i < s.length(); ++i)
  {
    if (s[i] == ' ')
    {
      s[i] = '0';
    }
  }

  return s;
}

// 功能: 取得當地現在時間, 格式為 microsecond
// 回傳值: 格式為 microsecond 的當地現在時間
// 沒有參數.
unsigned long long int mws_time::get_current_local_time_microsecond()
{
  timeval tim;
  gettimeofday(&tim, NULL);

  time_t raw_time = tim.tv_sec;
  struct tm time_info;
  localtime_TS(raw_time, time_info);

  unsigned long long int current_time_microsecond = ((unsigned long long int)time_info.tm_hour * 3600000000) +
                                                    ((unsigned long long int)time_info.tm_min * 60000000) +
                                                    ((unsigned long long int)time_info.tm_sec * 1000000) +
                                                    tim.tv_usec;

/*
  unsigned long long int current_time = timeval_to_local_time(tim);
  unsigned long long int current_time_microsecond = ((current_time / 10000000000) * 3600000000) +
                                                    (((current_time % 10000000000) / 100000000) * 60000000) +
                                                    (current_time % 100000000);
*/
  return current_time_microsecond;
}

// 功能: 將 timeval 轉換成 YYYYMMDD (日期)的數字
// 回傳值: YYYYMMDD 的 unsigned long long int
// 參數 tim: 要轉換的 timeval 格式時間.
unsigned long long int mws_time::timeval_to_local_date(const timeval &tim)
{
  time_t raw_time = tim.tv_sec;
  struct tm time_info;
  localtime_TS(raw_time, time_info);

  unsigned long long int local_date = (unsigned long long int)(time_info.tm_year + 1900) * 10000;
  local_date += ((unsigned long long int)(time_info.tm_mon + 1) * 100);
  local_date += time_info.tm_mday;

  return local_date;
}

// 功能: 將 timeval 轉換成兩位時兩位分兩位秒三位 millisecond 三位 microsecond 的數字
// 回傳值: 兩位時兩位分兩位秒三位 millisecond 三位 microsecond 的 unsigned long long int
// 參數 tim: 要轉換的 timeval 格式時間.
unsigned long long int mws_time::timeval_to_local_time(const timeval &tim)
{
  time_t raw_time = tim.tv_sec;
  struct tm time_info;
  localtime_TS(raw_time, time_info);

  unsigned long long int local_time = (unsigned long long int)time_info.tm_hour * 10000000000;
  local_time += ((unsigned long long int)time_info.tm_min * 100000000);
  local_time += ((unsigned long long int)time_info.tm_sec * 1000000);
  local_time += tim.tv_usec;

  return local_time;
}

// 功能: 判斷是否現在時間是否大於等於參數時間
// 回傳值: true 表示現在時間大於等於參數時間, false 表示現在時間小於參數時間
// 參數 hour: 幾點
// 參數 min: 幾分.
bool mws_time::current_time_is_euqal_to_or_later_than(const int hour,
                                                      const int min)
{
  time_t current_time = time(NULL);
  struct tm ts;
  localtime_TS(current_time, ts);

  if ((ts.tm_hour > hour) ||
      ((ts.tm_hour == hour) && (ts.tm_min >= min)))
  {
    return true;
  }

  return false;
}

// 功能: 判斷是否現在時間是否小於等於參數時間
// 回傳值: true 表示現在時間小於等於參數時間, false 表示現在時間大於參數時間
// 參數 hour: 幾點
// 參數 min: 幾分.
bool mws_time::current_time_is_euqal_to_or_earlier_than(const int hour,
                                                        const int min)
{
  time_t current_time = time(NULL);
  struct tm ts;
  localtime_TS(current_time, ts);

  if ((ts.tm_hour < hour) ||
      ((ts.tm_hour == hour) && (ts.tm_min <= min)))
  {
    return true;
  }

  return false;
}

// 功能: 進入本函式, 然後 delay 到參數時間離開
// 回傳值為 void
// 參數 hour: 幾點
// 參數 min: 幾分
// 參數 second: 幾秒.
void mws_time::delay_to_input_time(const int hour,
                                   const int min,
                                   const int second)
{
  time_t target_time = (time_t)((hour * 60 * 60) +
                                (min * 60) +
                                second);
  // 先取得標準時間.
  time_t current_time = time(NULL);
  // 將時間轉成 local time
  struct tm ts;
  localtime_TS(current_time, ts);

  // 用 local time 算出 current time 是今天的第幾秒.
  current_time = (time_t)((ts.tm_hour * 60 * 60) +
                          (ts.tm_min * 60) +
                          ts.tm_sec);
  while (target_time > current_time)
  {
    // 超過 300 秒, delay 240 秒.
    if ((target_time - current_time) > 300)
    {
      sleep(270);
    }
    // 超過 60 秒, delay 10 秒.
    else if ((target_time - current_time) > 30)
    {
      sleep(10);
    }
    else
    {
      sleep(1);
    }
    // 先取得標準時間.
    current_time = time(NULL);
    // 將時間轉成 local time
    localtime_TS(current_time, ts);
    // 用 local time 算出 current time 是今天的第幾秒.
    current_time = (time_t)((ts.tm_hour * 60 * 60) +
                            (ts.tm_min * 60) +
                            ts.tm_sec);
  }

  return ;
}
