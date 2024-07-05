#ifndef MWS_CLASS_DEFINITION_H_INCLUDED
#define MWS_CLASS_DEFINITION_H_INCLUDED

#include <netinet/in.h>
#include <arpa/inet.h>
#include <map>
#include <pthread.h>
#include <queue>
#include <sys/socket.h>
#include <string>
#include <netinet/tcp.h>
#include <vector>

#include "./mws_endianness.h"
#include "./mws_timer_callback.h"
#include "./mws_type_definition.h"

class mws_ctx_attr
{
  public:
    // 功能: context attribute 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式:
    //   pthread_stack_size: the minimum size (in bytes) that will
    //                       be allocated for select thread's creation.
    //     pthread_stack_size 的設定：
    //     1. 預設值 (會依環境改變, 以下參數為 2022/11/23 於測試系統取得)
    //       - linux: 8388608 bytes
    //       - NSK: 131072 bytes
    //     2. PTHREAD_STACK_MIN
    //       - linux: 16384 bytes
    //       - NSK: 4096 bytes (但實際上要大於等於 32768 bytes,
    //                          直接使用 PTHREAD_STACK_MIN 會失敗)
    //     3. 設定時值只要大於等於以下值就可以：
    //       - linux: 16384 bytes
    //       - NSK: 32768 bytes
    //     4. NSK 受保護的最小值為 PTHREAD_STACK_MIN_PROTECTED_STACK (49152)
    //        NSK 受保護的最大值為 PTHREAD_STACK_MAX_PROTECTED_STACK (16777216)
    //        NSK 不受保護的最大值為 PTHREAD_STACK_MAX_NP (33554432)
    mws_ctx_attr(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_ctx_attr();

    // 功能: 修改 context attribute 物件.
    // 回傳值: 沒有.
    // 參數: attr_name: attribute member 的名稱.
    // 參數: attr_value: attribute member 的值.
    void mws_modify_ctx_attr(std::string attr_name, std::string attr_value);

    friend class mws_ctx;
    friend int32_t mws_init_ctx(mws_ctx_t* ctx_ptr,
                                const bool is_from_cfg,
                                const mws_ctx_attr_t mws_ctx_attr,
                                const std::string cfg_section);
  private:
    std::string cfg_section;

    ssize_t pthread_stack_size;
};

// 預設擁有一個 src 跟一個 rcv 的 ctx.
class mws_ctx
{
  public:
    // ctx 的號碼.
    int ctx_no;

    // debug
    //ssize_t g_accu_recv_byte = 0;
    //bool is_debug_mode = false;
    // end debug

    // 功能: context 物件建構式.
    // 回傳值: 沒有.
    // 參數: mws_ctx_attr: context attribute 物件.
    mws_ctx(mws_ctx_attr_t mws_ctx_attr);

    // 功能: context 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式:
    //   pthread_stack_size: the minimum size (in bytes) that will
    //                       be allocated for select thread's creation.
    mws_ctx(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_ctx();

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -2: delay_usec > MAX_DELAY_USEC;
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       delay_usec: Delay until cb_function should be called (in microsecond(s)).
    //       is_recurring: Schedule a recurring timer that calls proc when it expires.
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               long delay_usec,
                               bool is_recurring);

    // 功能: Schedule a timer that calls callback function when it expires.
    //       (Total delay time as 'delay_sec + delay_usec').
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -1: delay_sec > MAX_DELAY_SEC;
    //         -2: delay_usec > MAX_DELAY_USEC;
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       delay_sec: Part of delay time (in second(s)).
    //       delay_usec: Part of delay time (in microsecond(s)).
    //       is_recurring: Schedule a recurring timer that calls proc when it expires.
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               long delay_sec,
                               long delay_usec,
                               bool is_recurring);

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       time_tv: Exact time in tmvl_t(timeval) format
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               tmvl_t time_tv);

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -1: Conversion failed - mktime() failed
    //         -2: year(>= 1900) or
    //             mon(1-12) or
    //             day(1-31) or
    //             hour(0-23) or
    //             min(0-59) or
    //             sec(0-61) or
    //             usec(0-999999)
    //             is out of range.
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       time_tv: Exact time in tmvl_t(timeval) format
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               int year,
                               int mon,
                               int mday,
                               int hour,
                               int min,
                               int sec,
                               int usec,
                               int isdst);

    // 功能: Cancel a previously scheduled timer identified by id.
    // 回傳值: 0: Timer cancelled.
    //         1: Timer does not exist(or no longer available).
    //         -1: timer_id >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       timer_id: The identifier specifying the timer to cancel.
    int32_t mws_cancel_timer(mws_evq_t* evq_ptr,
                             const int32_t timer_id);

    // 功能: Show version of this library.
    // 回傳值: Version infomation.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    std::string mws_timer_version(mws_evq_t* evq_ptr);

    // 功能: Show all timers' detail. (debug tool)
    // 回傳值: Number of timer(s).
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    int32_t mws_show_all_timer_detail(mws_evq_t* evq_ptr);

    // 功能: Show number of existing timer(s).
    // 回傳值: Number of timer(s).
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    int32_t mws_show_num_of_timer_with_evq(mws_evq_t* evq_ptr);

    // 功能: 回傳 ctx 的 cfg section name.
    // 回傳值: ctx 的 cfg section name.
    // 參數: 無.
    std::string mws_get_cfg_section();

    // 功能: 回傳 ctx 的物件狀態.
    // 回傳值: ctx 的物件狀態.
    //    MWS_NO_ERROR: 沒有錯誤.
    //    MWS_ERROR_PTHREAD_CREATE: 建立 pthread 發生錯誤.
    // 參數: 無.
    uint32_t mws_get_object_status();

    friend class mws_evq;
    friend class mws_src;
    friend class mws_rcv;
    friend int32_t mws_init_ctx(mws_ctx_t* ctx_ptr,
                                const bool is_from_cfg,
                                const mws_ctx_attr_t mws_ctx_attr,
                                const std::string cfg_section);
    friend int32_t mws_init_src(mws_src* src_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* src_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_src_attr_t mws_src_attr,
                                const std::string cfg_section);
    friend int32_t mws_init_rcv(mws_rcv* rcv_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* rcv_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_rcv_attr_t mws_rcv_attr,
                                const std::string cfg_section);
    friend void* ctx_thread_function(void* mws_ctx_ptr);
    friend int create_listen_socket(mws_src_t* src_ptr);
    friend int create_connect_socket(wait_to_connect_rcv_session_t& sess_info,
                                     sockaddr_in_t& rcv_listen_addr_info,
                                     sockaddr_in_t& rcv_conn_addr_info);
    friend void step_accept_connection(mws_ctx_t* ctx_ptr, fd_t selected_fd);
    friend int request_connection(mws_ctx_t* ctx_ptr, fd_t conn_fd);
    friend void src_conn_topic_check_error(std::deque<fd_t>::iterator& it,
                                           const std::string function,
                                           const int line_no);
    friend void step_src_conn_wait_fffd(std::deque<fd_t>::iterator& it);
    friend void step_src_conn_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void src_conn_ready_error(std::deque<fd_t>::iterator& it,
                                     const std::string function,
                                     const int line_no);
    friend void step_src_conn_ready(std::deque<fd_t>::iterator& it);
    friend void rcv_topic_check_error(std::deque<fd_t>::iterator& it,
                                      const std::string function,
                                      const int line_no);
    friend void step_rcv_wait_fefc(std::deque<fd_t>::iterator& it);
    friend void step_rcv_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void rcv_ready_error(std::deque<fd_t>::iterator& it,
                                const std::string function,
                                const int line_no);
    friend void step_rcv_ready(std::deque<fd_t>::iterator& it);
    friend void step_send_fe_error(std::deque<wait_to_check_topic_src_conn_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void step_send_ff_error(std::deque<wait_to_check_topic_rcv_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void step_rcv_connect(std::deque<wait_to_connect_rcv_session_t>::iterator& it,
                                 mws_ctx_t* ctx_ptr,
                                 const std::string function,
                                 const int line_no);
    friend void ctx_debug(mws_ctx_t* ctx_ptr,
                          const std::string function,
                          const int line_no,
                          bool show_ctx_list_wait_to_connect_rcv_session,
                          bool show_ctx_list_wait_to_check_topic_src_conn_session,
                          bool show_ctx_list_wait_to_check_topic_rcv_session,
                          bool show_ctx_list_wait_to_close_src_conn_fds,
                          bool show_ctx_list_wait_to_close_rcv_fds,
                          bool show_ctx_list_owned_src_listen_fds,
                          bool show_ctx_list_owned_src_conn_fds,
                          bool show_ctx_list_owned_rcv_fds,
                          bool show_all_set_and_max_fd);
    friend void rcv_topic_check_timeout_error(fd_t fd,
                                              const std::string function,
                                              const int line_no);
  private:
    uint32_t object_status;

    std::string cfg_section;

    // 可設定參數 begin.
    // ctx thread stack size.
    size_t pthread_stack_size;
    // 可設定參數 end.

    // 指向處理 timer callback function 的工具物件的指標.
    mws_timer_callback_t* timer_callback_ptr;

    // ctx thread 是否應該停止.
    bool must_stop_running_ctx_thread;
    // ctx thread 是否在運作中.
    bool is_ctx_thread_running;

    // 屬於這個 ctx 的 src.
    std::deque<mws_src_t*> ctx_list_owned_src;
    pthread_mutex_t ctx_list_owned_src_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_owned_src_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_owned_src_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_owned_src_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_owned_src_mutex_lock();
      int ctx_list_owned_src_mutex_trylock();
      void ctx_list_owned_src_mutex_unlock();
    #endif

    // 屬於這個 ctx 的 rcv.
    std::deque<mws_rcv_t*> ctx_list_owned_rcv;
    pthread_mutex_t ctx_list_owned_rcv_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_owned_rcv_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_owned_rcv_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_owned_rcv_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_owned_rcv_mutex_lock();
      int ctx_list_owned_rcv_mutex_trylock();
      void ctx_list_owned_rcv_mutex_unlock();
    #endif

    // 屬於這個 ctx 的 fd.
    fd_set all_set;
    // 屬於這個 ctx 的且在 select() 後有變動的 fd.
    fd_set rset;
    // 最大 fd 值.
    fd_t max_fd;
    // 功能: 依照輸入的 fd 和現在的 max_fd, 決定是否更新 max_fd.
    // 回傳值: 無.
    // 參數 fd: 新的 fd.
    void update_max_fd(const fd_t fd);

    // 等待 ctx 執行 connect() 指令連線到 src 的 rcv session 設定.
    std::deque<wait_to_connect_rcv_session_t> ctx_list_wait_to_connect_rcv_session;
    // 功能: 刪除 ctx_list_wait_to_connect_rcv_session 中, 屬於 rcv_ptr 的所有資料.
    // 回傳值: 沒有.
    // 參數 rcv_ptr: 指標指向要刪除的資料所屬的 rcv.
    void clear_data_of_specified_rcv_from_ctx_list_wait_to_connect_rcv_session(const mws_rcv_t* rcv_ptr);

    // 等待 ctx 執行 topic check 的 src conn session 設定.
    std::deque<wait_to_check_topic_src_conn_session_t> ctx_list_wait_to_check_topic_src_conn_session;
    // 功能: 刪除 ctx_list_wait_to_check_topic_src_conn_session 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_wait_to_check_topic_src_conn_session(const fd_t fd);

    // 等待 ctx 執行 topic check 的 rcv session 設定.
    std::deque<wait_to_check_topic_rcv_session_t> ctx_list_wait_to_check_topic_rcv_session;
    // 功能: 刪除 ctx_list_wait_to_check_topic_rcv_session 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_wait_to_check_topic_rcv_session(const fd_t fd);

    // 等待 ctx 執行 close() 的 src listen fds.
    std::deque<fd_t> ctx_list_wait_to_close_src_listen_fds;
    // 功能: 刪除 ctx_list_wait_to_close_src_listen_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_wait_to_close_src_listen_fds(const fd_t fd);

    // 等待 ctx 執行 close() 的 src conn fds.
    std::deque<fd_t> ctx_list_wait_to_close_src_conn_fds;
    // 功能: 刪除 ctx_list_wait_to_close_src_conn_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_wait_to_close_src_conn_fds(const fd_t fd);

    // 等待 ctx 執行 close() 的 rcv fds.
    std::deque<fd_t> ctx_list_wait_to_close_rcv_fds;
    // 功能: 刪除 ctx_list_wait_to_close_rcv_fdss 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_wait_to_close_rcv_fds(const fd_t fd);

    // 這個 ctx 所擁有的 src listen fds.
    std::deque<fd_t> ctx_list_owned_src_listen_fds;
    pthread_mutex_t ctx_list_owned_src_listen_fds_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_owned_src_listen_fds_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_owned_src_listen_fds_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_owned_src_listen_fds_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_owned_src_listen_fds_mutex_lock();
      int ctx_list_owned_src_listen_fds_mutex_trylock();
      void ctx_list_owned_src_listen_fds_mutex_unlock();
    #endif

    // 功能: 刪除 ctx_list_owned_src_listen_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_owned_src_listen_fds(const fd_t fd);

    // 這個 ctx 所擁有的 src conn fds.
    std::deque<fd_t> ctx_list_owned_src_conn_fds;
    pthread_mutex_t ctx_list_owned_src_conn_fds_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_owned_src_conn_fds_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_owned_src_conn_fds_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_owned_src_conn_fds_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_owned_src_conn_fds_mutex_lock();
      int ctx_list_owned_src_conn_fds_mutex_trylock();
      void ctx_list_owned_src_conn_fds_mutex_unlock();
    #endif

    // 功能: 刪除 ctx_list_owned_src_conn_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_owned_src_conn_fds(const fd_t fd);

    // 這個 ctx 所擁有的 rcv fds.
    std::deque<fd_t> ctx_list_owned_rcv_fds;
    pthread_mutex_t ctx_list_owned_rcv_fds_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_owned_rcv_fds_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_owned_rcv_fds_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_owned_rcv_fds_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_owned_rcv_fds_mutex_lock();
      int ctx_list_owned_rcv_fds_mutex_trylock();
      void ctx_list_owned_rcv_fds_mutex_unlock();
    #endif

    // 功能: 刪除 ctx_list_owned_rcv_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_ctx_list_owned_rcv_fds(const fd_t fd);

    // 等待 ctx 停止的 src 的指標.
    std::deque<mws_src_t*> ctx_list_wait_to_stop_src;
    pthread_mutex_t ctx_list_wait_to_stop_src_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_wait_to_stop_src_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_wait_to_stop_src_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_wait_to_stop_src_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_wait_to_stop_src_mutex_lock();
      int ctx_list_wait_to_stop_src_mutex_trylock();
      void ctx_list_wait_to_stop_src_mutex_unlock();
    #endif

    // 等待 ctx 停止的 rcv 的指標.
    std::deque<mws_rcv_t*> ctx_list_wait_to_stop_rcv;
    pthread_mutex_t ctx_list_wait_to_stop_rcv_mutex;
    #if (MWS_DEBUG == 1)
      void ctx_list_wait_to_stop_rcv_mutex_lock(const std::string file, const std::string function, const int line_no);
      int ctx_list_wait_to_stop_rcv_mutex_trylock(const std::string file, const std::string function, const int line_no);
      void ctx_list_wait_to_stop_rcv_mutex_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void ctx_list_wait_to_stop_rcv_mutex_lock();
      int ctx_list_wait_to_stop_rcv_mutex_trylock();
      void ctx_list_wait_to_stop_rcv_mutex_unlock();
    #endif

    // select thread ID.
    pthread_t ctx_thread_id;

    // Begin: private member function.
    // 功能: checks if any scheduled callback function has expired.
    //       if so, executes those functions.
    // 回傳值: 無.
    // 參數: 無.
    void timer_manager();

    // End: private member function.
};

class mws_reactor_only_ctx_attr
{
  public:
    // 功能: reactor only context attribute 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式:
    //   pthread_stack_size: the minimum size (in bytes) that will
    //                       be allocated for thread's creation.
    //     pthread_stack_size 的設定：
    //     1. 預設值 (會依環境改變, 以下參數為 2022/11/23 於測試系統取得)
    //       - linux: 8388608 bytes
    //       - NSK: 131072 bytes
    //     2. PTHREAD_STACK_MIN
    //       - linux: 16384 bytes
    //       - NSK: 4096 bytes (但實際上要大於 32768 bytes,
    //                          直接使用 PTHREAD_STACK_MIN 會失敗)
    //     3. 設定時值只要大於以下值就可以：
    //       - linux: 16384 bytes
    //       - NSK: 32768 bytes
    //     4. NSK 受保護的最小值為 PTHREAD_STACK_MIN_PROTECTED_STACK (49152)
    //        NSK 受保護的最大值為 PTHREAD_STACK_MAX_PROTECTED_STACK (16777216)
    //        NSK 不受保護的最大值為 PTHREAD_STACK_MAX_NP (33554432)
    mws_reactor_only_ctx_attr(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_reactor_only_ctx_attr();

    // 功能: 修改 reactor only context attribute 物件.
    // 回傳值: 沒有.
    // 參數: attr_name: attribute member 的名稱.
    // 參數: attr_value: attribute member 的值.
    void mws_modify_reactor_only_ctx_attr(std::string attr_name,
                                          std::string attr_value);

    friend class mws_reactor_only_ctx;

  private:
    std::string cfg_section;

    ssize_t pthread_stack_size;
};

class mws_reactor_only_ctx
{
  public:
    // 功能: reactor only context 物件建構式.
    // 回傳值: 沒有.
    // 參數: mws_reactor_only_ctx_attr: reactor only context attribute 物件.
    mws_reactor_only_ctx(mws_reactor_only_ctx_attr_t mws_reactor_only_ctx_attr);

    // 功能: reactor only context 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式:
    //   pthread_stack_size: the minimum size (in bytes) that will
    //                       be allocated for thread's creation.
    mws_reactor_only_ctx(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_reactor_only_ctx();

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -2: delay_usec > MAX_DELAY_USEC;
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       delay_usec: Delay until cb_function should be called (in microsecond(s)).
    //       is_recurring: Schedule a recurring timer that calls proc when it expires.
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               long delay_usec,
                               bool is_recurring);

    // 功能: Schedule a timer that calls callback function when it expires.
    //       (Total delay time as 'delay_sec + delay_usec').
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -1: delay_sec > MAX_DELAY_SEC;
    //         -2: delay_usec > MAX_DELAY_USEC;
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       delay_sec: Part of delay time (in second(s)).
    //       delay_usec: Part of delay time (in microsecond(s)).
    //       is_recurring: Schedule a recurring timer that calls proc when it expires.
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               long delay_sec,
                               long delay_usec,
                               bool is_recurring);

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       time_tv: Exact time in tmvl_t(timeval) format
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               tmvl_t time_tv);

    // 功能: Schedule a timer that calls callback function when it expires.
    // 回傳值: >= 0: Timer ID (successful completion).
    //         -1: Conversion failed - mktime() failed
    //         -2: year(>= 1900) or
    //             mon(1-12) or
    //             day(1-31) or
    //             hour(0-23) or
    //             min(0-59) or
    //             sec(0-61) or
    //             usec(0-999999)
    //             is out of range.
    //         -3: Number of timer >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       cb_function: The function to call when the timer expires.
    //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
    //       time_tv: Exact time in tmvl_t(timeval) format
    int32_t mws_schedule_timer(mws_evq_t* evq_ptr,
                               timer_callback_t cb_function,
                               void* custom_data_ptr,
                               int year,
                               int mon,
                               int mday,
                               int hour,
                               int min,
                               int sec,
                               int usec,
                               int isdst);

    // 功能: Cancel a previously scheduled timer identified by id.
    // 回傳值: 0: Timer cancelled.
    //         1: Timer does not exist(or no longer available).
    //         -1: timer_id >= MAX_TIMER_NUM.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    //       timer_id: The identifier specifying the timer to cancel.
    int32_t mws_cancel_timer(mws_evq_t* evq_ptr,
                             const int32_t timer_id);

    // 功能: Show version of this library.
    // 回傳值: Version infomation.
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    std::string mws_timer_version(mws_evq_t* evq_ptr);

    // 功能: Show all timers' detail. (debug tool)
    // 回傳值: Number of timer(s).
    // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
    int32_t mws_show_all_timer_detail(mws_evq_t* evq_ptr);

    // 功能: 回傳 reactor only ctx 的 cfg section name.
    // 回傳值: reactor only ctx 的 cfg section name.
    // 參數: 無.
    std::string mws_get_cfg_section();

    // 功能: 回傳 reactor only ctx 的物件狀態.
    // 回傳值: reactor only ctx 的物件狀態.
    //    MWS_NO_ERROR: 沒有錯誤.
    //    MWS_ERROR_PTHREAD_CREATE: 建立 pthread 發生錯誤.
    // 參數: 無.
    uint32_t mws_get_object_status();

    friend void* reactor_only_ctx_thread_function(void* mws_ctx_ptr);

  private:
    uint32_t object_status;

    std::string cfg_section;

    // reactor only ctx thread 是否應該停止.
    bool must_stop_running_ctx_thread;
    // reactor only ctx thread 是否在運作中.
    bool is_ctx_thread_running;

    // thread ID.
    pthread_t ctx_thread_id;

    // 指向處理 timer callback function 的工具物件的指標.
    mws_timer_callback_t* timer_callback_ptr;

    // 功能: checks if any scheduled callback function has expired.
    //       if so, executes those functions.
    // 回傳值: 無.
    // 參數: 無.
    void timer_manager();
};

class mws_evq_attr
{
  public:
    // 功能: event queue attribute 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式: is_auto_dispatch:
    //                true: auto dispatch.
    //                false: manual dispatch.
    mws_evq_attr(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_evq_attr();

    // 功能: 修改 event queue attribute 物件.
    // 回傳值: 沒有.
    // 參數: attr_name: attribute member 的名稱.
    // 參數: attr_value: attribute member 的值.
    void mws_modify_evq_attr(std::string attr_name, std::string attr_value);

    friend class mws_evq;
    friend int32_t mws_init_evq(mws_evq_t* evq_ptr,
                                const bool is_from_cfg,
                                const mws_evq_attr_t mws_evq_attr,
                                const std::string cfg_section);
  private:
    std::string cfg_section;

    // true: auto dispatch.
    // false: manual dispatch.
    bool is_auto_dispatch;
};

class mws_evq
{
  public:
    // 功能: event queue 物件建構式.
    // 回傳值: 沒有.
    // 參數: mws_evq_attr: event queue attribute 物件.
    mws_evq(mws_evq_attr_t mws_evq_attr);

    // 功能: event queue 物件建構式.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式: is_auto_dispatch:
    //                true: auto dispatch.
    //                false: manual dispatch.
    mws_evq(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_evq();

    // 功能: dispatch 該 event queue 的 event.
    // 回傳值:    0: 正常.
    //         非 0: 異常/失敗 (可由 AP programmer 在 callback function 中定義).
    // 參數: 無.
    int mws_event_dispatch();

    // 功能: 回傳 evq 的 cfg section name.
    // 回傳值: evq 的 cfg section name.
    // 參數: 無.
    std::string mws_get_cfg_section();

    // 功能: 回傳 evq 的物件狀態.
    // 回傳值: evq 的物件狀態.
    //    MWS_NO_ERROR: 沒有錯誤.
    // 參數: 無.
    uint32_t mws_get_object_status();

    friend class mws_reactor_only_ctx;
    friend class mws_ctx;
    friend class mws_src;
    friend class mws_rcv;
    friend int32_t mws_init_evq(mws_evq_t* evq_ptr,
                                const bool is_from_cfg,
                                const mws_evq_attr_t mws_evq_attr,
                                const std::string cfg_section);
    friend void* ctx_thread_function(void* mws_ctx_ptr);
    friend int create_listen_socket(mws_src_t* src_ptr);
    friend int create_connect_socket(wait_to_connect_rcv_session_t& sess_info,
                                     sockaddr_in_t& rcv_listen_addr_info,
                                     sockaddr_in_t& rcv_conn_addr_info);
    friend void step_accept_connection(mws_ctx_t* ctx_ptr, fd_t selected_fd);
    friend void step_src_conn_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void step_rcv_connect(std::deque<wait_to_connect_rcv_session_t>::iterator& it,
                                 mws_ctx_t* ctx_ptr,
                                 const std::string function,
                                 const int line_no);
    friend void step_send_fe_error(std::deque<wait_to_check_topic_src_conn_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void step_send_ff_error(std::deque<wait_to_check_topic_rcv_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void step_rcv_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void src_conn_topic_check_error(std::deque<fd_t>::iterator& it,
                                           const std::string function,
                                           const int line_no);
    friend void src_conn_ready_error(std::deque<fd_t>::iterator& it,
                                     const std::string function,
                                     const int line_no);
    friend void rcv_topic_check_error(std::deque<fd_t>::iterator& it,
                                      const std::string function,
                                      const int line_no);
    friend void rcv_ready_error(std::deque<fd_t>::iterator& it,
                                const std::string function,
                                const int line_no);
    friend void step_src_conn_ready(std::deque<fd_t>::iterator& it);
    friend void step_rcv_ready(std::deque<fd_t>::iterator& it);
    friend void ctx_debug(mws_ctx_t* ctx_ptr,
                          const std::string function,
                          const int line_no,
                          bool show_ctx_list_wait_to_connect_rcv_session,
                          bool show_ctx_list_wait_to_check_topic_src_conn_session,
                          bool show_ctx_list_wait_to_check_topic_rcv_session,
                          bool show_ctx_list_wait_to_close_src_conn_fds,
                          bool show_ctx_list_wait_to_close_rcv_fds,
                          bool show_ctx_list_owned_src_listen_fds,
                          bool show_ctx_list_owned_src_conn_fds,
                          bool show_ctx_list_owned_rcv_fds,
                          bool show_all_set_and_max_fd);
    friend void rcv_topic_check_timeout_error(fd_t fd,
                                              const std::string function,
                                              const int line_no);
  private:
    uint32_t object_status;

    std::string cfg_section;

    // true: auto dispatch.
    // false: manual dispatch.
    bool is_auto_dispatch;
    bool is_dispatch_thread_running;

    // evq 的號碼, 和指向 evq 的指標合併為 evq_id.
    int evq_no;

    // 指向處理 timer callback function 的工具物件的指標.
    mws_timer_callback_t* timer_callback_ptr;

    // 維護這個 evq 資料時使用.
    pthread_mutex_t mut_data_maintain;

    #if (MWS_DEBUG == 1)
      void evq_lock(const std::string file, const std::string function, const int line_no);
      int evq_trylock(const std::string file, const std::string function, const int line_no);
      void evq_unlock(const std::string file, const std::string function, const int line_no);
    #else
      void evq_lock();
      int evq_trylock();
      void evq_unlock();
    #endif

    // 決定是否讓 dispatch thread 放行.
    pthread_mutex_t mut_select_done;
    pthread_cond_t cond_select_done;
    #if (MWS_DEBUG == 1)
      void evq_cond_lock(const std::string file, const std::string function, const int line_no);
      void evq_cond_unlock(const std::string file, const std::string function, const int line_no);
      void evq_cond_wait(const std::string file, const std::string function, const int line_no);
    #else
      void evq_cond_lock();
      void evq_cond_unlock();
      void evq_cond_wait();
    #endif

    // 是否應該呼叫 pthread_cond_signal().
    bool flag_must_unlock;

    // 放置待處理的 MWS_SRC_EVENT_CONNECT, MWS_MSG_BOS events 的 queue.
    std::queue<mws_event_t*> connect_event_queue;
    // 放置待處理的 MWS_SRC_EVENT_DISCONNECT, MWS_MSG_EOS events 的 queue.
    std::queue<mws_event_t*> disconnect_event_queue;

    // 這個 evq 所擁有的 src conn fds 和 rcv fds.
    std::deque<fd_t> evq_list_owned_fds;
    // 功能: 刪除 evq_list_owned_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_evq_list_owned_fds(const fd_t fd);

    // Begin: private member function.
    mws_event_t* create_non_msg_event(fd_t fd, uint8_t event_type, bool need_to_lock_fd);
    ssize_t push_back_non_msg_event(mws_event_t* event_ptr);

    int dispatch_connect_events();
    int dispatch_disconnect_events();
    int dispatch_events();

    // 功能: checks if any scheduled callback function has expired.
    //       if so, executes those functions.
    // 回傳值: 無.
    // 參數: 無.
    void timer_manager();
    // End: private member function.

    #if (MWS_DEBUG == 1)
      time_t prev_check_time;
    #endif
};

class mws_src_attr
{
  public:
    // 功能: source attribute 物件建構式, 讀取 cfg 資料設定物件參數, 如果讀取 cfg 資料失敗, 以預設值設定物件參數.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式: topic_name: 關注的 topic name.
    //              listen_ip: 此 src 的 ip.
    //              listen_port: 此 src 的 port.
    //              is_hot_failover_recv_mode:
    //                true: hot failover 接收資料模式,
    //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    mws_src_attr(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_src_attr();

    // 功能: 修改 source attribute 物件.
    // 回傳值: 沒有.
    // 參數: attr_name: attribute member 的名稱.
    // 參數: attr_value: attribute member 的值.
    void mws_modify_src_attr(std::string attr_name, std::string attr_value);

    friend class mws_src;
    friend int32_t mws_init_src(mws_src* src_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* src_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_src_attr_t mws_src_attr,
                                const std::string cfg_section);

  private:
    std::string cfg_section;

    std::string topic_name;

    // true: hot failover 接收資料模式,
    //       message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    // false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    bool is_hot_failover_recv_mode;

    ip_port_low_high_t src_ip_port;
};

class mws_src
{
  public:
    // 功能: source 物件建構式.
    // 回傳值: 沒有.
    // 參數: mws_src_attr: source attribute 物件.
    //       ctx_ptr: 屬於哪個 ctx.
    //       evq_ptr: 使用的 evq.
    //       src_cb: event 執行的 callback function.
    //       custom_data_ptr: callback function 的引數 (optional).
    //       custom_data_size: callback function 的引數大小 (byte) (optional).
    mws_src(mws_src_attr_t mws_src_attr,
            mws_ctx_t* ctx_ptr,
            mws_evq_t* evq_ptr,
            callback_t* src_cb_ptr,
            void* custom_data_ptr = NULL,
            const size_t custom_data_size = 0);

    // 功能: source 物件建構式, 讀取 cfg 資料設定物件參數, 如果讀取 cfg 資料失敗, 以預設值設定物件參數.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    //       ctx_ptr: 屬於哪個 ctx.
    //       evq_ptr: 使用的 evq.
    //       src_cb: event 執行的 callback function.
    //       custom_data_ptr: callback function 的引數 (optional).
    //       custom_data_size: callback function 的引數大小 (byte) (optional).
    // config 格式: topic_name: 關注的 topic name.
    //              listen_ip: 此 src 的 ip.
    //              listen_port: 此 src 的 port.
    //              is_hot_failover_recv_mode:
    //                true: hot failover 接收資料模式,
    //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    mws_src(std::string cfg_section,
            mws_ctx_t* ctx_ptr,
            mws_evq_t* evq_ptr,
            callback_t* src_cb_ptr,
            void* custom_data_ptr = NULL,
            const size_t custom_data_size = 0);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_src();

    // 功能: send a message from source to receiver(s).
    // 回傳值:
    //   ok 表示 fd 成功送出資料而且沒有 queue 在 deque.
    //   block 表示 fd 的資料有 queue 在 deque 但沒發生底層錯誤.
    //   fail 表示 fd 會斷線.
    //   0: 沒有 fail + 沒有 block + 發生 ok (全部 fd 傳送完成)
    //   1: 沒有 fail + 沒有 block + 沒有 ok (沒有接收端)
    //   2: 沒有 fail + 發生 block + 發生 ok
    //   3: 沒有 fail + 發生 block + 沒有 ok
    //   4: 發生 fail + 沒有 block + 發生 ok
    //   5: 發生 fail + 沒有 block + 沒有 ok (全部 fd 斷線)
    //   6: 發生 fail + 發生 block + 發生 ok
    //   7: 發生 fail + 發生 block + 沒有 ok
    //  -1: len 超過 MAX_MSG_SIZE.
    // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
    // 參數 size_t len: 要傳送的 message 的長度 (byte).
    // 參數 flags: 決定以 flush 或是 nonblock 模式送出訊息.
    int mws_src_send(const char* msg_ptr,
                     size_t len,
                     int flags = MWS_SEND_MSG_FLUSH);

    // 功能: send a message from source to receiver(s) with sequence-checking (hot failover mode).
    // 回傳值:
    //   ok 表示 fd 成功送出資料而且沒有 queue 在 deque.
    //   block 表示 fd 的資料有 queue 在 deque 但沒發生底層錯誤.
    //   fail 表示 fd 會斷線.
    //   0: 沒有 fail + 沒有 block + 發生 ok (全部 fd 傳送完成)
    //   1: 沒有 fail + 沒有 block + 沒有 ok (沒有接收端)
    //   2: 沒有 fail + 發生 block + 發生 ok
    //   3: 沒有 fail + 發生 block + 沒有 ok
    //   4: 發生 fail + 沒有 block + 發生 ok
    //   5: 發生 fail + 沒有 block + 沒有 ok (全部 fd 斷線)
    //   6: 發生 fail + 發生 block + 發生 ok
    //   7: 發生 fail + 發生 block + 沒有 ok
    //  -1: len 超過 MAX_MSG_SIZE.
    // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
    // 參數 size_t len: 要傳送的 message 的長度 (byte).
    // 參數 seq_num: hot failover send 時所使用的序號.
    // 參數 flags: 決定以 flush 或是 nonblock 模式送出訊息.
    int mws_hf_src_send(const char* msg_ptr,
                        size_t len,
                        uint64_t seq_num,
                        int flags = MWS_SEND_MSG_FLUSH);

    // 功能: 回傳 src 的 cfg section name.
    // 回傳值: src 的 cfg section name.
    // 參數: 無.
    std::string mws_get_cfg_section();

    // 功能: 回傳 src 的物件狀態.
    // 回傳值: src 的物件狀態.
    //    MWS_NO_ERROR: 沒有錯誤.
    //    MWS_ERROR_TOPIC_NAME: topic name 錯誤.
    //    MWS_ERROR_LISTEN_SOCKET_CREATE: 建立 listen socket 發生錯誤.
    // 參數: 無.
    uint32_t mws_get_object_status();

    // 功能: 回傳 src 是否為 hot failover 接收資料模式.
    // 回傳值: src 是否為 hot failover 接收資料模式.
    // 參數: 無.
    bool mws_is_hot_failover_recv_mode();

    // 功能: 回傳 src 的 topic name.
    // 回傳值: src 的 topic name.
    // 參數: 無.
    std::string get_topic_name();

    friend class mws_ctx;
    friend class mws_evq;
    friend int32_t mws_init_src(mws_src* src_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* src_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_src_attr_t mws_src_attr,
                                const std::string cfg_section);
    friend void* ctx_thread_function(void* mws_ctx_ptr);
    friend int create_listen_socket(mws_src_t* src_ptr);
    friend void step_accept_connection(mws_ctx_t* ctx_ptr, fd_t selected_fd);
    friend void fd_info_log(const fd_t fd, std::string& log_body);
    friend void src_conn_topic_check_error(std::deque<fd_t>::iterator& it,
                                           const std::string function,
                                           const int line_no);
    friend void step_src_conn_wait_fffd(std::deque<fd_t>::iterator& it);
    friend void step_src_conn_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void src_conn_ready_error(std::deque<fd_t>::iterator& it,
                                     const std::string function,
                                     const int line_no);
    friend void step_src_conn_ready(std::deque<fd_t>::iterator& it);
    friend void step_send_fe_error(std::deque<wait_to_check_topic_src_conn_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void ctx_debug(mws_ctx_t* ctx_ptr,
                          const std::string function,
                          const int line_no,
                          bool show_ctx_list_wait_to_connect_rcv_session,
                          bool show_ctx_list_wait_to_check_topic_src_conn_session,
                          bool show_ctx_list_wait_to_check_topic_rcv_session,
                          bool show_ctx_list_wait_to_close_src_conn_fds,
                          bool show_ctx_list_wait_to_close_rcv_fds,
                          bool show_ctx_list_owned_src_listen_fds,
                          bool show_ctx_list_owned_src_conn_fds,
                          bool show_ctx_list_owned_rcv_fds,
                          bool show_all_set_and_max_fd);
    friend void update_g_fd_table_status(const fd_t fd,
                                         const int16_t new_status,
                                         const std::string function,
                                         const int line_no);
  private:
    uint32_t object_status;

    std::string cfg_section;

    // 可設定參數 begin.
    // topic name.
    std::string topic_name;
    // 接收資料模式.
    // true: hot failover 接收資料模式,
    //       message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    // false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    bool is_hot_failover_recv_mode;
    // source 的 IP 和 port range, 用以連線.
    ip_port_low_high_t src_ip_port;
    // 可設定參數 end.

    mws_ctx_t* ctx_ptr;
    mws_evq_t* evq_ptr;
    // 記錄當前 sequence number (src 為 hot failover 接收資料模式時使用).
    uint64_t max_seq_num;
    // callback 函式.
    callback_t* cb_ptr;

    // data for callback function.
    void* custom_data_ptr;
    // custom_data_ptr 內容的長度.
    size_t custom_data_size;

    // bind 正確完成後得到的 listen address information.
    sockaddr_in_t src_listen_addr;
    // listen socket 正確建立後得到的 fd.
    fd_t src_listen_fd;
    // 每個 accept 正確完成後得到的 src connect fd,
    // 在傳送資料時使用(為了求快不使用 deque), 在 fd 錯誤時清除.
    std::vector<fd_t> src_connect_fds;
    // 功能: 刪除 src_connect_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_src_connect_fds(fd_t fd);

    // 功能: src send message 發生錯誤時的後續處理.
    // 回傳值: 沒有.
    // 參數 fd: 發生錯誤的 src conn fd.
    // 參數 function: 呼叫的 function, 要傳入 __func__.
    // 參數 line_no: 呼叫的行號, 要傳入 __LINE__.
    void src_send_error(fd_t fd,
                        const std::string function,
                        const int line_no);

    // 功能: send a heartbeat message from source to receiver(s).
    // 回傳值: 沒有.
    // 參數: 無.
    void mws_src_send_heartbeat();

    // 是否已經準備好可以解構 src.
    bool flag_ready_to_release_src;
};

class mws_rcv_attr
{
  public:
    // 功能: receiver attribute 物件建構式, 讀取 cfg 資料設定物件參數, 如果讀取 cfg 資料失敗, 以預設值設定物件參數.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    // config 格式: topic_name: 關注的 topic name.
    //              sess_addr_pair_XX: 此 rcv 所負責的所有 session 的 socket (listen & connect) 組合, XX 從01到99.
    //              is_hot_failover_recv_mode:
    //                true: hot failover 接收資料模式,
    //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_MSG_DATA event.
    //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_MSG_DATA event.
    mws_rcv_attr(std::string cfg_section);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_rcv_attr();

    // 功能: 修改 receiver attribute 物件.
    // 回傳值: 沒有.
    // 參數: attr_name: attribute member 的名稱.
    // 參數: attr_value: attribute member 的值.
    void mws_modify_rcv_attr(std::string attr_name, std::string attr_value);

    friend class mws_rcv;
    friend int32_t mws_init_rcv(mws_rcv* rcv_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* rcv_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_rcv_attr_t mws_rcv_attr,
                                const std::string cfg_section);
  private:
    std::string cfg_section;

    std::string topic_name;

    // true: hot failover 接收資料模式,
    //       message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_MSG_DATA event.
    // false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_MSG_DATA event.
    bool is_hot_failover_recv_mode;

    size_t num_of_rcv_sessions;
    sess_addr_pair_t rcv_session_list[MAX_FD_SIZE];
};

class mws_rcv
{
  public:
    // 功能: receiver 物件建構式.
    // 回傳值: 沒有.
    // 參數: mws_rcv_attr: receiver attribute 物件.
    //       ctx_ptr: 屬於哪個 ctx.
    //       evq_ptr: 使用的 evq.
    //       rcv_cb: event 執行的 callback function.
    //       custom_data_ptr: callback function 的引數 (optional).
    //       custom_data_size: callback function 的引數大小 (byte) (optional).
    mws_rcv(mws_rcv_attr_t mws_rcv_attr,
            mws_ctx_t* ctx_ptr,
            mws_evq_t* evq_ptr,
            callback_t* rcv_cb_ptr,
            void* custom_data_ptr = NULL,
            const size_t custom_data_size = 0);

    // 功能: receiver 物件建構式, 讀取 cfg 資料設定物件參數, 如果讀取 cfg 資料失敗, 以預設值設定物件參數.
    // 回傳值: 沒有.
    // 參數: cfg_section: config 的 section name.
    //       ctx_ptr: 屬於哪個 ctx.
    //       evq_ptr: 使用的 evq.
    //       rcv_cb: event 執行的 callback function.
    //       custom_data_ptr: callback function 的引數 (optional).
    //       custom_data_size: callback function 的引數大小 (byte) (optional).
    // config 格式: topic_name: 關注的 topic name.
    //              sess_addr_pair_XX: 此 rcv 所負責的所有 session 的 socket (listen & connect) 組合, XX 從01到99.
    //              is_hot_failover_recv_mode:
    //                true: hot failover 接收資料模式,
    //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_MSG_DATA event.
    //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_MSG_DATA event.
    mws_rcv(std::string cfg_section,
            mws_ctx_t* ctx_ptr,
            mws_evq_t* evq_ptr,
            callback_t* rcv_cb_ptr,
            void* custom_data_ptr = NULL,
            const size_t custom_data_size = 0);

    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_rcv();

    // 功能: send a message from receiver to source(s)
    // 回傳值:
    //   ok 表示 fd 成功送出資料而且沒有 queue 在 deque.
    //   block 表示 fd 的資料有 queue 在 deque 但沒發生底層錯誤.
    //   fail 表示 fd 會斷線.
    //   0: 沒有 fail + 沒有 block + 發生 ok (全部 fd 傳送完成)
    //   1: 沒有 fail + 沒有 block + 沒有 ok (沒有接收端)
    //   2: 沒有 fail + 發生 block + 發生 ok
    //   3: 沒有 fail + 發生 block + 沒有 ok
    //   4: 發生 fail + 沒有 block + 發生 ok
    //   5: 發生 fail + 沒有 block + 沒有 ok (全部 fd 斷線)
    //   6: 發生 fail + 發生 block + 發生 ok
    //   7: 發生 fail + 發生 block + 沒有 ok
    //  -1: len 超過 MAX_MSG_SIZE.
    // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
    // 參數 size_t len: 要傳送的 message 的長度 (byte).
    // 參數 flags: 決定以 flush 或是 nonblock 模式送出訊息.
    int mws_rcv_send(const char* msg_ptr,
                     size_t len,
                     int flags = MWS_SEND_MSG_FLUSH);

    // 功能: send a message from receiver to source(s) with sequence-checking (hot failover mode).
    // 回傳值:
    //   ok 表示 fd 成功送出資料而且沒有 queue 在 deque.
    //   block 表示 fd 的資料有 queue 在 deque 但沒發生底層錯誤.
    //   fail 表示 fd 會斷線.
    //   0: 沒有 fail + 沒有 block + 發生 ok (全部 fd 傳送完成)
    //   1: 沒有 fail + 沒有 block + 沒有 ok (沒有接收端)
    //   2: 沒有 fail + 發生 block + 發生 ok
    //   3: 沒有 fail + 發生 block + 沒有 ok
    //   4: 發生 fail + 沒有 block + 發生 ok
    //   5: 發生 fail + 沒有 block + 沒有 ok (全部 fd 斷線)
    //   6: 發生 fail + 發生 block + 發生 ok
    //   7: 發生 fail + 發生 block + 沒有 ok
    //  -1: len 超過 MAX_MSG_SIZE.
    // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
    // 參數 size_t len: 要傳送的 message 的長度 (byte).
    // 參數 seq_num: hot failover send 時所使用的序號.
    // 參數 flags: 決定以 flush 或是 nonblock 模式送出訊息.
    int mws_hf_rcv_send(const char* msg_ptr,
                        size_t len,
                        uint64_t seq_num,
                        int flags = MWS_SEND_MSG_FLUSH);

    // 功能: 回傳 rcv 的 cfg section name.
    // 回傳值: rcv 的 cfg section name.
    // 參數: 無.
    std::string mws_get_cfg_section();

    // 功能: 回傳 rcv 的物件狀態.
    // 回傳值: rcv 的物件狀態.
    //    MWS_NO_ERROR: 沒有錯誤.
    //    MWS_ERROR_TOPIC_NAME: topic name 錯誤.
    //    MWS_ERROR_CONNECT_SOCKET_CREATE: 建立 connect socket 發生錯誤.
    // 參數: 無.
    uint32_t mws_get_object_status();

    // 功能: 回傳 rcv 是否為 hot failover 接收資料模式.
    // 回傳值: rcv 是否為 hot failover 接收資料模式.
    // 參數: 無.
    bool mws_is_hot_failover_recv_mode();

    // 功能: 回傳 rcv 的 session pair 數量.
    // 回傳值: rcv 的 session pair 數量.
    // 參數: 無.
    size_t mws_get_num_of_rcv_sessions();

    // 功能: 回傳 rcv 的 topic name.
    // 回傳值: rcv 的 topic name.
    // 參數: 無.
    std::string get_topic_name();

    friend class mws_ctx;
    friend class mws_evq;
    friend int32_t mws_init_rcv(mws_rcv* rcv_ptr,
                                mws_ctx_t* ctx_ptr,
                                mws_evq_t* evq_ptr,
                                callback_t* rcv_cb_ptr,
                                void* custom_data_ptr,
                                const size_t custom_data_size,
                                const bool is_from_cfg,
                                const mws_rcv_attr_t mws_rcv_attr,
                                const std::string cfg_section);
    friend void* ctx_thread_function(void* mws_ctx_ptr);
    friend int create_connect_socket(wait_to_connect_rcv_session_t& sess_info,
                                     sockaddr_in_t& rcv_listen_addr_info,
                                     sockaddr_in_t& rcv_conn_addr_info);
    friend int request_connection(mws_ctx_t* ctx_ptr, fd_t conn_fd);
    friend void fd_info_log(const fd_t fd, std::string& log_body);
    friend void rcv_topic_check_error(std::deque<fd_t>::iterator& it,
                                      const std::string function,
                                      const int line_no);
    friend void step_rcv_wait_fefc(std::deque<fd_t>::iterator& it);
    friend void step_rcv_wait_topic_name(std::deque<fd_t>::iterator& it);
    friend void rcv_ready_error(std::deque<fd_t>::iterator& it,
                                const std::string function,
                                const int line_no);
    friend void step_rcv_ready(std::deque<fd_t>::iterator& it);
    friend void step_send_ff_error(std::deque<wait_to_check_topic_rcv_session_t>::iterator& it,
                                   const std::string function,
                                   const int line_no);
    friend void step_rcv_connect(std::deque<wait_to_connect_rcv_session_t>::iterator& it,
                                 mws_ctx_t* ctx_ptr,
                                 const std::string function,
                                 const int line_no);
    friend void ctx_debug(mws_ctx_t* ctx_ptr,
                          const std::string function,
                          const int line_no,
                          bool show_ctx_list_wait_to_connect_rcv_session,
                          bool show_ctx_list_wait_to_check_topic_src_conn_session,
                          bool show_ctx_list_wait_to_check_topic_rcv_session,
                          bool show_ctx_list_wait_to_close_src_conn_fds,
                          bool show_ctx_list_wait_to_close_rcv_fds,
                          bool show_ctx_list_owned_src_listen_fds,
                          bool show_ctx_list_owned_src_conn_fds,
                          bool show_ctx_list_owned_rcv_fds,
                          bool show_all_set_and_max_fd);
    friend void update_g_fd_table_status(const fd_t fd,
                                         const int16_t new_status,
                                         const std::string function,
                                         const int line_no);
    friend void rcv_topic_check_timeout_error(fd_t fd,
                                              const std::string function,
                                              const int line_no);
  private:
    uint32_t object_status;

    std::string cfg_section;

    // 可設定參數 begin.
    // topic name.
    std::string topic_name;
    // 接收資料模式.
    // true: hot failover 接收資料模式,
    //       message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    // false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    bool is_hot_failover_recv_mode;
    // receiver 的 IP 和 port range 以及要連線的 source 的 IP 和 port range, 用以連線.
    sess_addr_pair_t rcv_session_list[MAX_FD_SIZE];
    // 有幾組 rcv_session_list.
    size_t num_of_rcv_sessions;
    // 可設定參數 end.

    mws_ctx_t* ctx_ptr;
    mws_evq_t* evq_ptr;
    // 記錄當前 sequence number (src 為 hot failover 接收資料模式時使用).
    uint64_t max_seq_num;
    // callback 函式.
    callback_t* cb_ptr;

    // data for callback function.
    void* custom_data_ptr;
    // custom_data_ptr 內容的長度.
    size_t custom_data_size;

    // fd management =====
    // bind 正確完成後得到的 connect address information.
    //sockaddr_in_t rcv_connect_addr[MAX_FD_SIZE];
    // bind 正確完成後得到的 listen address information.
    //sockaddr_in_t rcv_listen_addr[MAX_FD_SIZE];
    // 每個 connect 正確完成後得到的 connect fd,
    // 在 fd 錯誤時清除.
    // 在傳送資料時使用(為了求快不使用 deque), 在斷線時清除.
    std::vector<fd_t> rcv_connect_fds;
    // 功能: 刪除 rcv_connect_fds 中, fd 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 fd 的資料.
    // 參數 fd: 要刪除的資料的 fd.
    int erase_rcv_connect_fds(fd_t fd);
    // fd management =====

    // 用來記錄已使用的 src address, 避免重複連線.
    std::deque<sockaddr_in_t> rcv_list_connected_src_address;
    // 功能: 刪除 rcv_list_connected_src_address 中, socket address info 等於傳入值的資料.
    // 回傳值 0: 正確完成刪除資料.
    // 回傳值 1: 沒有該 socket address info 的資料.
    // 參數 addr_info: 要刪除的資料的 socket address info.
    int erase_rcv_list_connected_src_address(sockaddr_in_t addr_info);

    // 功能: rcv send message 發生錯誤時的後續處理.
    // 回傳值: 沒有.
    // 參數 fd: 發生錯誤的 rcv fd.
    // 參數 function: 呼叫的 function, 要傳入 __func__.
    // 參數 line_no: 呼叫的行號, 要傳入 __LINE__.
    void rcv_send_error(fd_t fd,
                        const std::string function,
                        const int line_no);

    // 功能: send a heartbeat message from receiver to source(s).
    // 回傳值: 沒有.
    // 參數: 無.
    void mws_rcv_send_heartbeat();

    // 是否已經準備好可以解構 rcv.
    bool flag_ready_to_release_rcv;
};

#endif // MWS_CLASS_DEFINITION_H_INCLUDED
