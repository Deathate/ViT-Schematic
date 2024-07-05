#ifndef MWS_TIMER_CALLBACK_H_INCLUDED
#define MWS_TIMER_CALLBACK_H_INCLUDED

#include <sys/time.h>
#include <sys/types.h>

#include <pthread.h>

#include <stdint.h>
#include <map>
#include <string>

//--------------------------------------------------------------------
// * Timer Callback Library Notice:
// *  *** Callback functioon format:
// *        int (*FUNCTION_NAME)(void*)
// *  *** User defined data type 'tmvl_t' format:
// *        struct timeval
// *        {
// *          long tv_sec;
// *          long tv_usec;
// *        }¡F
// *  *** User defined data type 'tm_t' format:
// *        struct tm
// *        {
// *          int tm_sec;
// *          int tm_min;
// *          int tm_hour;
// *          int tm_mday;
// *          int tm_mon;
// *          int tm_year;
// *          int tm_wday;
// *          int tm_yday;
// *          int tm_isdst;
// *        }¡F
// *  *** Max timer number: 10,000.
// *  *** Max delay time: 100,010.000000 seconds.
// *        second(s): 100,000.
// *        +
// *        microsecond(s): 10,000,000 (10 seconds).
// *  *** Timer ID:
// *        Type: int32_t.
// *        Value range: 0 to (MAX_TIMER_NUM - 1)
// *  *** Time table's key format:
// *        struct _ttb_key_t
// *        {
// *          tmvl_t tmvl_val;
// *          uint32_t seq_no = 0;
// *        };
//--------------------------------------------------------------------

// Max number of timer is 500.
#define MAX_TIMER_NUM 500
// Max delay time is 100,000 seconds.
#define MAX_DELAY_SEC 100000
// Max delay time is 10,000,000 microseconds (10 seconds).
#define MAX_DELAY_USEC 10000000

typedef struct timeval tmvl_t;
typedef struct tm tm_t;

class mws_timer_callback;
typedef mws_timer_callback mws_timer_callback_t;
typedef int (*timer_callback_t)(void*);

// Key of time table.
struct _ttb_key_t
{
  tmvl_t tmvl_val;
  uint32_t seq_no = 0;
};
typedef _ttb_key_t ttb_key_t;

// Record whether a time ID is being used and its corresponding time table key.
struct _id_status_t
{
  ttb_key_t key;
  bool is_used = false;
};
typedef _id_status_t id_status_t;

// For comparing key values of time table.
// * Return value:
// *   0: x >= y.
// *   1: x < y.
template <class T>
struct Compare
{
  int operator()(const T& x, const T& y) const
  {
    if ((x.tmvl_val.tv_sec > y.tmvl_val.tv_sec) ||
        ((x.tmvl_val.tv_sec == y.tmvl_val.tv_sec) &&
         (x.tmvl_val.tv_usec > y.tmvl_val.tv_usec)) ||
        ((x.tmvl_val.tv_sec == y.tmvl_val.tv_sec) &&
         (x.tmvl_val.tv_usec == y.tmvl_val.tv_usec) &&
         (x.seq_no >= y.seq_no)))
    {
      return 0;
    }
    else
    {
      return 1;
    }
  }
};

// A schedule datum (Timer).
// tmvl_val: Expiry time.
// cb: The function to call when the timer expires.
// custom_data_ptr: Pointer to client data that is passed when the timer expires.
// delay_usec: Delay until cb_function should be called (in microseconds).
// id: The identifier specifying the timer to cancel.
// is_recurring: Schedule a recurring timer that calls proc when it expires.
struct _schedule_datum
{
  tmvl_t tmvl_val;
  timer_callback_t cb;
  void* custom_data_ptr = NULL;
  long delay_sec = 0;
  long delay_usec = 0;
  int32_t id = -1;
  bool is_recurring = false;
};
typedef _schedule_datum schedule_datum_t;

class mws_timer_callback
{
  public:
    // * Constructor.
    // * Argument:
    //     auto_management:
    //       true: Automated management(Constructor will create a thread).
    //       false: Manual management(Constructor will not create any thread).
    // * Return value: No return type.
    mws_timer_callback(bool auto_management = true);

    // * Destructor.
    // * Argument: None.
    // * Return value: No return type.
    ~mws_timer_callback();

    // * Schedule a timer that calls callback function when it expires.
    // * Argument:
    //     cb_function: The function to call when the timer expires.
    //     clientd_ptr: Pointer to client data that is passed when the timer expires.
    //     delay_usec: Delay until cb_function should be called (in microsecond(s)).
    //                 (Max delay_usec is 10,000,000 microseconds (10 seconds)).
    //     is_recurring: Schedule a recurring timer that calls proc when it expires.
    // * Return value:
    //     >= 0: Timer ID (successful completion).
    //     -2: delay_usec > MAX_DELAY_USEC;
    //     -3: Number of timer >= MAX_TIMER_NUM.
    int32_t schedule_timer(timer_callback_t cb_function,
                           void* custom_data_ptr,
                           long delay_usec,
                           bool is_recurring);

    // * Schedule a timer that calls callback function when it expires.
    // * Argument:
    //     cb_function: The function to call when the timer expires.
    //     clientd_ptr: Pointer to client data that is passed when the timer expires.
    //     delay_sec: Part of delay time (in second(s)).
    //                (Max delay_sec is 100,000 seconds).
    //     delay_usec: Part of delay time (in microsecond(s)).
    //                 (Max delay_usec is 10,000,000 microseconds (10 seconds)).
    //     is_recurring: Schedule a recurring timer that calls proc when it expires.
    // * Return value:
    //     >= 0: Timer ID (successful completion).
    //     -1: delay_sec > MAX_DELAY_SEC;
    //     -2: delay_usec > MAX_DELAY_USEC;
    //     -3: Number of timer >= MAX_TIMER_NUM.
    // * Notice: Total delay time is 'delay_sec + delay_usec'.
    int32_t schedule_timer(timer_callback_t cb_function,
                           void* custom_data_ptr,
                           long delay_sec,
                           long delay_usec,
                           bool is_recurring);

    // * Schedule a timer that calls callback function when it expires.
    // * Argument:
    //     cb_function: The function to call when the timer expires.
    //     clientd_ptr: Pointer to client data that is passed when the timer expires.
    //     time_tv: Exact time in tmvl_t(timeval) format
    // * Return value:
    //     >= 0: Timer ID (successful completion).
    //     -3: Number of timer >= MAX_TIMER_NUM.
    int32_t schedule_timer(timer_callback_t cb_function,
                           void* custom_data_ptr,
                           tmvl_t time_tv);

    // * Schedule a timer that calls callback function when it expires.
    // * Argument:
    //     cb_function: The function to call when the timer expires.
    //     clientd_ptr: Pointer to client data that is passed when the timer expires.
    //     time_tv: Exact time in tmvl_t(timeval) format
    // * Return value:
    //     >= 0: Timer ID (successful completion).
    //     -1: Conversion failed - mktime() failed
    //     -2: year(>= 1900) or
    //         mon(1-12) or
    //         day(1-31) or
    //         hour(0-23) or
    //         min(0-59) or
    //         sec(0-61) or
    //         usec(0-999999)
    //         is out of range.
    //     -3: Number of timer >= MAX_TIMER_NUM.
    int32_t schedule_timer(timer_callback_t cb_function,
                           void* custom_data_ptr,
                           int year,
                           int mon,
                           int mday,
                           int hour,
                           int min,
                           int sec,
                           int usec,
                           int isdst);

    // * Cancel a previously scheduled timer identified by id.
    // * Argument:
    //     timer_id: The identifier specifying the timer to cancel.
    // * return value:
    //     0: Timer cancelled.
    //     1: Timer does not exist(or no longer available).
    //     -1: timer_id >= MAX_TIMER_NUM.
    int32_t cancel_timer(const int32_t timer_id);

    // * Timer callback manager.
    // * Argument: None.
    // * Return value: None.
    void timer_manager();

    // * Show version of this library.
    // * Argument: None.
    // * Return value: Version infomation.
    std::string version();

    // * Show all timers' detail. (debug tool)
    // * Argument: None.
    // * return value: Number of timer(s).
    int32_t show_all_timer_detail();

    // * Show number of existing timer(s).
    // * Argument: None.
    // * return value: Number of timer(s).
    int32_t show_num_of_timer();

    friend void* tmcb_fun(void* tmcb_ptr);

  private:
    // true: Automated management(Constructor will create a thread).
    // false: Manual management(Constructor will not create any thread).
    bool auto_management = true;
    // Number of existing timer(s).
    int32_t existing_timer_cnt = 0;
    // Last timer ID.
    int32_t last_timer_id = 0;
    // 'stop_thread_flag == true' means stop thread 'thread_tmcb'.
    bool stop_thread_flag = false;
    // Index is timer ID, content is timer ID's status.
    id_status_t id_status[MAX_TIMER_NUM];
    // Time table key - schedule datum pair & key compare operator.
    std::map<ttb_key_t, schedule_datum_t, Compare<ttb_key_t> > time_table;
    // Thread for timer callback manager.
    pthread_t thread_tmcb;
#ifndef TCB_NOT_THREAD_SAFE
    // Mutual exclusion.
    pthread_mutex_t mutex;
#endif
};

#endif
