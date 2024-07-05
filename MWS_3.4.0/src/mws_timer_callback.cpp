//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_TIMER_CALLBACK_CPP 1

#ifndef TCB_NOT_THREAD_SAFE
  #define TIMER_CALLBACK_LIB_VERSION "3.0.0 (Thread-safe Mode)"
#else
  #define TIMER_CALLBACK_LIB_VERSION "3.0.0 (Not thread-safe Mode)"
#endif

#include <sys/time.h>
#include <sys/types.h>

#include <pthread.h>
#include <unistd.h>

#include <stdint.h>
#include <string.h>
#include <iomanip>
#include <iostream>

#include <map>
#include <string>

#include "../inc/mws_timer_callback.h"
#include "../inc/mws_log.h"

void* tmcb_fun(void*);

int32_t convert_time_string_to_tmvl(tmvl_t& result_tv,
                                    int year,
                                    int mon,
                                    int mday,
                                    int hour,
                                    int min,
                                    int sec,
                                    int usec,
                                    int isdst);

using namespace mws_log;

mws_timer_callback::mws_timer_callback(bool auto_management)
{
  // Initialize member variables.
  this->auto_management = auto_management;
  this->existing_timer_cnt = 0;
  this->last_timer_id = 0;
  this->stop_thread_flag = false;
  //memset(this->id_status, 0x0, sizeof(this->id_status));
  //memset(this->id_status, 0x0, (sizeof(id_status_t) * MAX_TIMER_NUM));
  for (size_t i = 0; i < MAX_TIMER_NUM; ++i)
  {
    this->id_status[i].key.tmvl_val.tv_sec = 0;
    this->id_status[i].key.tmvl_val.tv_usec = 0;
    this->id_status[i].key.seq_no = 0;
    this->id_status[i].is_used = false;
  }
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_init(&(this->mutex),NULL);
#endif

  if (this->auto_management == true)
  {
    // Create pthread for managing time table.
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    int rtv = pthread_create(&(this->thread_tmcb),
                             &attr,
                             tmcb_fun,
                             (void*)this);
    if (rtv != 0)
    {
      std::string log_body = "pthread_create() failed (rtv: " + std::to_string(rtv)
                             + ", errno: " + std::to_string(errno)
                             + ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      //std::cerr << "file: " << __FILE__ << " "
      //          << "fun: " << __func__ << " "
      //          << "ln: " << __LINE__ << " "
      //          << "pthread_create() failed, "
      //          << "errno = " << rtv
      //          << std::endl;
      exit(EXIT_FAILURE);
    }

    pthread_attr_destroy(&attr);
  }

  return;
}

mws_timer_callback::~mws_timer_callback()
{
  if (this->auto_management == true)
  {
    this->stop_thread_flag = true;
    pthread_join(this->thread_tmcb, NULL);
  }
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_destroy(&(this->mutex));
#endif
  return;
}

int32_t mws_timer_callback::schedule_timer(timer_callback_t cb_function,
                                           void* custom_data_ptr,
                                           long delay_usec,
                                           bool is_recurring)
{
  if (delay_usec > (long)MAX_DELAY_USEC)
  {
    return -2;
  }
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif
  if (this->existing_timer_cnt >= (int32_t)MAX_TIMER_NUM)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return -3;
  }

  // get timer ID.
  ++this->last_timer_id;
  if (last_timer_id >= MAX_TIMER_NUM)
  {
    this->last_timer_id = 0;
  }
  while ((this->id_status[this->last_timer_id]).is_used == true)
  {
    ++this->last_timer_id;
    if (last_timer_id >= MAX_TIMER_NUM)
    {
      this->last_timer_id = 0;
    }
  }

  // Calculate expiry time.
  tmvl_t tv;
  gettimeofday(&tv, NULL);
  tv.tv_usec += delay_usec;
  tv.tv_sec += (tv.tv_usec / 1000000);
  tv.tv_usec %= 1000000;

  // Key of time_table map.
  ttb_key_t new_key;
  new_key.tmvl_val = tv;
  new_key.seq_no = 0;

  // Value of time_table map.
  schedule_datum_t new_timer;
  new_timer.tmvl_val = tv;
  new_timer.cb = cb_function;
  new_timer.custom_data_ptr = custom_data_ptr;
  new_timer.delay_sec = 0;
  new_timer.delay_usec = delay_usec;
  new_timer.id = this->last_timer_id;
  new_timer.is_recurring = is_recurring;

  // Maintain time table.
  std::pair<std::map<ttb_key_t, schedule_datum_t>::iterator,bool> ret;
  ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  while (ret.second == false)
  {
    ++new_key.seq_no;
    ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  }

//  while (this->time_table.find(new_key) != this->time_table.end())
//  {
//    ++new_key.seq_no;
//  }
//  this->time_table[new_key] = new_timer;

  // Maintain id_status.
  (this->id_status[this->last_timer_id]).key = new_key;
  (this->id_status[this->last_timer_id]).is_used = true;

  // Count existing timer.
  ++(this->existing_timer_cnt);

  int32_t timer_id = this->last_timer_id;

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return timer_id;
}

int32_t mws_timer_callback::schedule_timer(timer_callback_t cb_function,
                                           void* custom_data_ptr,
                                           long delay_sec,
                                           long delay_usec,
                                           bool is_recurring)
{
  if (delay_sec > (long)MAX_DELAY_SEC)
  {
    return -1;
  }

  if (delay_usec > (long)MAX_DELAY_USEC)
  {
    return -2;
  }

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif

  if (this->existing_timer_cnt >= (int32_t)MAX_TIMER_NUM)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return -3;
  }

  // get timer ID.
  ++this->last_timer_id;
  if (last_timer_id >= MAX_TIMER_NUM)
  {
    this->last_timer_id = 0;
  }
  while ((this->id_status[this->last_timer_id]).is_used == true)
  {
    ++this->last_timer_id;
    if (last_timer_id >= MAX_TIMER_NUM)
    {
      this->last_timer_id = 0;
    }
  }

  // Calculate expiry time.
  tmvl_t tv;
  gettimeofday(&tv, NULL);
  tv.tv_sec += delay_sec;
  tv.tv_usec += delay_usec;
  tv.tv_sec += (tv.tv_usec / 1000000);
  tv.tv_usec %= 1000000;

  // Key of time_table map.
  ttb_key_t new_key;
  new_key.tmvl_val = tv;
  new_key.seq_no = 0;
  // Value of time_table map.
  schedule_datum_t new_timer;
  new_timer.tmvl_val = tv;
  new_timer.cb = cb_function;
  new_timer.custom_data_ptr = custom_data_ptr;
  new_timer.delay_sec = delay_sec;
  new_timer.delay_usec = delay_usec;
  new_timer.id = this->last_timer_id;
  new_timer.is_recurring = is_recurring;

  // Maintain time table.
  std::pair<std::map<ttb_key_t, schedule_datum_t>::iterator,bool> ret;
  ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  while (ret.second == false)
  {
    ++new_key.seq_no;
    ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  }

//  while (this->time_table.find(new_key) != this->time_table.end())
//  {
//    ++new_key.seq_no;
//  }
//  this->time_table[new_key] = new_timer;

  // Maintain id_status.
  (this->id_status[this->last_timer_id]).key = new_key;
  (this->id_status[this->last_timer_id]).is_used = true;

  // Count existing timer.
  ++(this->existing_timer_cnt);

  int32_t timer_id = this->last_timer_id;

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return timer_id;
}

int32_t mws_timer_callback::schedule_timer(timer_callback_t cb_function,
                                           void* custom_data_ptr,
                                           tmvl_t time_tv)
{
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif

  if (this->existing_timer_cnt >= (int32_t)MAX_TIMER_NUM)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return -3;
  }

  // get timer ID.
  ++this->last_timer_id;
  if (last_timer_id >= MAX_TIMER_NUM)
  {
    this->last_timer_id = 0;
  }
  while ((this->id_status[this->last_timer_id]).is_used == true)
  {
    ++this->last_timer_id;
    if (last_timer_id >= MAX_TIMER_NUM)
    {
      this->last_timer_id = 0;
    }
  }

  // Key of time_table map.
  ttb_key_t new_key;
  new_key.tmvl_val = time_tv;
  new_key.seq_no = 0;

  // Value of time_table map.
  schedule_datum_t new_timer;
  new_timer.tmvl_val = time_tv;
  new_timer.cb = cb_function;
  new_timer.custom_data_ptr = custom_data_ptr;
  new_timer.delay_sec = 0;
  new_timer.delay_usec = 0;
  new_timer.id = this->last_timer_id;
  new_timer.is_recurring = false;

  // Maintain time table.
  std::pair<std::map<ttb_key_t, schedule_datum_t>::iterator,bool> ret;
  ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  while (ret.second == false)
  {
    ++new_key.seq_no;
    ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  }

//  while (this->time_table.find(new_key) != this->time_table.end())
//  {
//    ++new_key.seq_no;
//  }
//  this->time_table[new_key] = new_timer;

  // Maintain id_status.
  (this->id_status[this->last_timer_id]).key = new_key;
  (this->id_status[this->last_timer_id]).is_used = true;

  // Count existing timer.
  ++(this->existing_timer_cnt);

  int32_t timer_id = this->last_timer_id;

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return timer_id;
}

int32_t mws_timer_callback::schedule_timer(timer_callback_t cb_function,
                                           void* custom_data_ptr,
                                           int year,
                                           int mon,
                                           int mday,
                                           int hour,
                                           int min,
                                           int sec,
                                           int usec,
                                           int isdst)
{
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif

  if (this->existing_timer_cnt >= (int32_t)MAX_TIMER_NUM)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return -3;
  }

  tmvl_t time_tv;
  int32_t rtv = convert_time_string_to_tmvl(time_tv,
                                            year,
                                            mon,
                                            mday,
                                            hour,
                                            min,
                                            sec,
                                            usec,
                                            isdst);
  if (rtv != 0)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return rtv;
  }

  // get timer ID.
  ++this->last_timer_id;
  if (last_timer_id >= MAX_TIMER_NUM)
  {
    this->last_timer_id = 0;
  }
  while ((this->id_status[this->last_timer_id]).is_used == true)
  {
    ++this->last_timer_id;
    if (last_timer_id >= MAX_TIMER_NUM)
    {
      this->last_timer_id = 0;
    }
  }

  // Key of time_table map.
  ttb_key_t new_key;
  new_key.tmvl_val = time_tv;
  new_key.seq_no = 0;

  // Value of time_table map.
  schedule_datum_t new_timer;
  new_timer.tmvl_val = time_tv;
  new_timer.cb = cb_function;
  new_timer.custom_data_ptr = custom_data_ptr;
  new_timer.delay_sec = 0;
  new_timer.delay_usec = 0;
  new_timer.id = this->last_timer_id;
  new_timer.is_recurring = false;

  // Maintain time table.
  std::pair<std::map<ttb_key_t, schedule_datum_t>::iterator,bool> ret;
  ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  while (ret.second == false)
  {
    ++new_key.seq_no;
    ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
  }

//  while (this->time_table.find(new_key) != this->time_table.end())
//  {
//    ++new_key.seq_no;
//  }
//  this->time_table[new_key] = new_timer;

  // Maintain id_status.
  (this->id_status[this->last_timer_id]).key = new_key;
  (this->id_status[this->last_timer_id]).is_used = true;

  // Count existing timer.
  ++(this->existing_timer_cnt);

  int32_t timer_id = this->last_timer_id;

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return timer_id;
}

int32_t mws_timer_callback::cancel_timer(const int32_t timer_id)
{
  if (timer_id >= (int32_t)MAX_TIMER_NUM)
  {
    return -1;
  }

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif

  if ((this->id_status[timer_id]).is_used == false)
  {
#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif
    return 1;
  }

  int32_t rtv = 0;

  ttb_key_t key = (this->id_status[timer_id]).key;

  std::string::size_type ret = this->time_table.erase(key);
  // Cancel specified timer successfully.
  if (ret == 1)
  {
    --(this->existing_timer_cnt);
    (this->id_status[timer_id]).is_used = false;
    rtv = 0;
  }
  // Error: data inconsistent.
  else
  {
    std::string log_body = "cancel_timer() id_status and time_table are inconsistent";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    //std::cerr << "file: " << __FILE__ << " "
    //          << "fun: " << __func__ << " "
    //          << "ln: " << __LINE__ << " "
    //          << "id_status and time_table are inconsistent."
    //          << std::endl;
    exit(EXIT_FAILURE);
  }

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return rtv;
}

// * Body of timer callback manager.
// * Argument: None.
// * Return value: None.
void mws_timer_callback::timer_manager()
{
  if (!(this->time_table).empty())
  {
    // Get current time.
    tmvl_t tv;

    gettimeofday(&tv, NULL);

#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_lock(&(this->mutex));
#endif

    // Check each timer and run callback function if time is up.
    for (std::map<ttb_key_t, schedule_datum_t, Compare<ttb_key_t> >::iterator it = (this->time_table).begin(), next_it = it;
         it != (this->time_table).end();
         it = next_it)
    {
      ++next_it;
      // Compare timer and current time,
      // if timer <= current time, call assigned callback function.
      if ((it->second.tmvl_val.tv_sec < tv.tv_sec) ||
          ((it->second.tmvl_val.tv_sec == tv.tv_sec) && (it->second.tmvl_val.tv_usec <= tv.tv_usec)))
      {
        // Call assigned function.
        int rtv = (*(it->second.cb))(it->second.custom_data_ptr);
        if (rtv != 0)
        {
          std::string log_body = "timer_manager() call callback function failed";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

          //std::cerr << "file: " << __FILE__ << " "
          //          << "fun: " << __func__ << " "
          //          << "ln: " << __LINE__ << " "
          //          << "call callback function failed, "
          //          << "errno = " << rtv
          //          << std::endl;
          exit(EXIT_FAILURE);
        }
        if (it->second.is_recurring == false)
        {
          // The task of this timer has been completed.
          --(this->existing_timer_cnt);
          (this->id_status[it->second.id]).is_used = false;
          this->time_table.erase(it);
        }
        else
        {
          // Using the same timer ID
          // to create a new timer(another expiry time) and
          // erase original one(the task of this timer has been completed).

          // Get new time.
          tmvl_t new_tv = tv;

          new_tv.tv_sec += it->second.delay_sec;
          new_tv.tv_usec += it->second.delay_usec;

          new_tv.tv_sec += (new_tv.tv_usec / 1000000);
          new_tv.tv_usec %= 1000000;
          // Key of time_table map.
          ttb_key_t new_key;
          new_key.tmvl_val = new_tv;
          new_key.seq_no = 0;

          // Value of time_table map.
          schedule_datum_t new_timer;
          new_timer.tmvl_val = new_tv;
          new_timer.cb = it->second.cb;
          new_timer.custom_data_ptr = it->second.custom_data_ptr;
          new_timer.delay_sec = it->second.delay_sec;
          new_timer.delay_usec = it->second.delay_usec;
          new_timer.id = it->second.id;
          new_timer.is_recurring = it->second.is_recurring;

          // Maintain time table.
          std::pair<std::map<ttb_key_t, schedule_datum_t>::iterator,bool> ret;
          ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
          while (ret.second == false)
          {
            ++new_key.seq_no;
            ret = this->time_table.insert(std::pair<ttb_key_t, schedule_datum_t>(new_key, new_timer));
          }
//        while (this->time_table.find(new_key) != this->time_table.end())
//        {
//          ++new_key.seq_no;
//        }
//        this->time_table[new_key] = new_timer;

          // Maintain id_status.
          (this->id_status[it->second.id]).key = new_key;
          //(this->id_status[it->second.id]).is_used = true;
          this->time_table.erase(it);
        }
      }
      else
      {
        // Expiry time of remaining timer(s) > current time.
        break;
      }
    }

#ifndef TCB_NOT_THREAD_SAFE
    pthread_mutex_unlock(&(this->mutex));
#endif

  }

  return ;
}

std::string mws_timer_callback::version()
{
  return std::string(TIMER_CALLBACK_LIB_VERSION);
}

int32_t mws_timer_callback::show_all_timer_detail()
{
#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_lock(&(this->mutex));
#endif

  std::cout << "--id_status-----------------------------------------------------" << std::endl;

  std::cout << "id_status:" << std::endl;
  int32_t id_used_cnt = 0;
  for (int32_t i = 0; i < (int32_t)MAX_TIMER_NUM; ++i)
  {
    if (id_status[i].is_used == true)
    {
      ++id_used_cnt;
      std::cout << "ID:" << i << ", "
                << "key:" << id_status[i].key.tmvl_val.tv_sec << " "
                          << id_status[i].key.tmvl_val.tv_usec << " "
                          << id_status[i].key.seq_no
                          << std::endl;
    }
  }

  std::cout << "--time_table----------------------------------------------------" << std::endl;

  int32_t timer_cnt = 0;

  for (std::map<ttb_key_t, schedule_datum_t, Compare<ttb_key_t> >::iterator it = (this->time_table).begin();
       it != (this->time_table).end();
       ++it, ++timer_cnt)
  {
    // Show ID, callback function address, expiry time,
    // custom data address, delay time, recurring.
    std::cout << "key:" << it->first.tmvl_val.tv_sec << " "
                        << it->first.tmvl_val.tv_usec << " "
                        << it->first.seq_no << ", "
              << "expiry:" << it->second.tmvl_val.tv_sec << "."
                           << std::setw(6) << std::setfill('0') << it->second.tmvl_val.tv_usec << ", "
              << "CB:" << std::hex << (long*)(it->second.cb) << std::dec << ", "
              << "custom data:" << std::hex << (long*)(it->second.custom_data_ptr) << std::dec << ", "
              << "delay:" << it->second.delay_sec << "." << it->second.delay_usec << ", "
              << "ID:" << it->second.id << ", "
              << "recurring:" << ((it->second.is_recurring == true) ? "Y" : "N" )
              << std::endl;
  }

  if (timer_cnt != this->existing_timer_cnt)
  {
    std::string log_body = "show_all_timer_detail() existing_timer_cnt is incorrect";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    //std::cerr << "file: " << __FILE__ << " "
    //          << "fun: " << __func__ << " "
    //          << "ln: " << __LINE__ << " "
    //          << "existing_timer_cnt is incorrect, "
    //          << "existing_timer_cnt = " << existing_timer_cnt << ", "
    //          << "timer_cnt = " << timer_cnt
    //          << std::endl;
    exit(EXIT_FAILURE);
  }

  if (id_used_cnt != this->existing_timer_cnt)
  {
    std::string log_body = "show_all_timer_detail() existing_timer_cnt and id_used_cnt are inconsistent";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    //std::cerr << "file: " << __FILE__ << " "
    //          << "fun: " << __func__ << " "
    //          << "ln: " << __LINE__ << " "
    //          << "existing_timer_cnt and id_used_cnt are inconsistent, "
    //          << "existing_timer_cnt = " << existing_timer_cnt << ", "
    //          << "id_used_cnt = " << id_used_cnt
    //          << std::endl;
    exit(EXIT_FAILURE);
  }

#ifndef TCB_NOT_THREAD_SAFE
  pthread_mutex_unlock(&(this->mutex));
#endif

  return timer_cnt;
}

int32_t mws_timer_callback::show_num_of_timer()
{
  return this->existing_timer_cnt;
}

// * Body(thread) for executing timer callback manager.
// * Argument:
//     tmcb_ptr: Pointer to mws_timer_callback object.
// * Return value:
//     NULL: Leaving this callback function.
void* tmcb_fun(void* tmcb_void_ptr)
{
  mws_timer_callback_t* tmcb_ptr = ((mws_timer_callback_t*)tmcb_void_ptr);

  while (tmcb_ptr->stop_thread_flag == false)
  {
    tmcb_ptr->timer_manager();
    sched_yield();
  }

  return NULL;
}

// Convert time from calendar format to timeval format
// Parameters:
//     &result_tv: time of timeval format
//     year: >= 1900, local time
//     mon: month, range 1 to 12, 1 means January, local time
//     mday: day of the month, range 1 to 31, local time
//     hour: hours, range 0 to 23, local time
//     min: minutes, range 0 to 59, local time
//     sec: seconds, range 0 to 61, local time
//     usec: microseconds, range 0 to 999999, local time
//     isdst: daylight saving time, > 0: in effect, = 0: not in effect, < 0: not available.
//     time difference: time zone, range UTC-12:00 - UTC+14:00
//                      format: +/-hhmm
//                      e.g. UTC +8:00 = 800 or + 800
//                           UTC -3:45 = -345
// Returnd value:
//      0: Conversion succeeded
//     -1: Conversion failed - mktime() failed
//     -2: year or
//         mon or
//         day or
//         hour or
//         min or
//         sec or
//         usec
//         is out of range
int32_t convert_time_string_to_tmvl(tmvl_t& result_tv,
                                    int year,
                                    int mon,
                                    int mday,
                                    int hour,
                                    int min,
                                    int sec,
                                    int usec,
                                    int isdst)
{
  if ((year < 1900) ||
      (mon < 1) || (mon > 12) ||
      (mday < 1) || (mday > 31) ||
      (hour < 0) || (hour > 23) ||
      (min < 0) || (min > 59) ||
      (sec < 0) || (sec > 61) ||
      (usec < 0) || (usec > 999999))
  {
    std::string log_body = "convert_time_string_to_tmvl() error";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    //std::cerr << "convert_time_string_to_tmvl error:" << std::endl;
    //std::cerr << "  year: " << year << std::endl;
    //std::cerr << "  mon: " << mon << std::endl;
    //std::cerr << "  mday: " << mday << std::endl;
    //std::cerr << "  hour: " << hour << std::endl;
    //std::cerr << "  min: " << min << std::endl;
    //std::cerr << "  sec: " << sec << std::endl;
    //std::cerr << "  usec: " << usec << std::endl;
    return -2;
  }

  tm_t tm_obj;
  tm_obj.tm_sec = sec; // seconds,  range 0 to 61
  tm_obj.tm_min = min; // minutes, range 0 to 59
  tm_obj.tm_hour = hour; // hours, range 0 to 23
  tm_obj.tm_mday = mday; // day of the month, range 1 to 31
  tm_obj.tm_mon = mon - 1; // month, range 0 to 11, 0 means January
  tm_obj.tm_year = year - 1900; // The number of years since 1900, 0 means A.D. 1900
  //tm_obj.tm_wday; // day of the week, range 0 to 6, 0 means Sunday
  //tm_obj.tm_yday; // day in the year, range 0 to 365, 0 means the first of January
  tm_obj.tm_isdst = isdst; // daylight saving time, > 0: in effect, = 0: not in effect, < 0: not available.

  result_tv.tv_sec = mktime(&tm_obj);
  if (result_tv.tv_sec < 0)
  {
    return -1;
  }

  result_tv.tv_usec = usec;

  return 0;
}
