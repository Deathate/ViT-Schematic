#ifndef MWS_GLOBAL_VARIABLE_H_
#define MWS_GLOBAL_VARIABLE_H_

#include <deque>
#include <map>
#include <string>

#include "./mws_class_definition.h"
#include "./mws_endianness.h"
#include "./mws_type_definition.h"

namespace mws_global_variable
{
  #if defined(MWS_INIT_CPP)
    // MWS config 檔案路徑.
    std::string g_cfg_file_path;
  #endif

  #if defined(MWS_INIT_CPP)
    // MWS config 內容.
    std::map<std::string, std::map<std::string, std::string> > g_config_mapping;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_EVQ_CPP) || \
         defined(MWS_RCV_CPP) || \
         defined(MWS_REACTOR_ONLY_CTX_CPP) || \
         defined(MWS_SRC_CPP))
    extern std::map<std::string, std::map<std::string, std::string> > g_config_mapping;
  #endif

  #if defined(MWS_INIT_CPP)
    // MWS log 檔案路徑.
    std::string g_mws_log_file_path;
  #endif

  #if defined(MWS_INIT_CPP)
    // 寫 mws log 的等級.
    // 0 表示只寫必須的 log 訊息.
    // 1 表示寫入 debug 用的訊息.
    int16_t g_mws_log_level;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_EVQ_CPP) || \
         defined(MWS_RCV_CPP) || \
         defined(MWS_SOCKET_CPP) || \
         defined(MWS_SRC_CPP) || \
         defined(MWS_UTIL_CPP))
    extern int16_t g_mws_log_level;
  #endif

  #if defined(MWS_INIT_CPP)
    // 建立過的 ctx 數量.
    int g_num_of_ctx;
  #elif (defined(MWS_CTX_CPP))
    extern int g_num_of_ctx;
  #endif

  #if defined(MWS_INIT_CPP)
    // 建立過的 evq 數量.
    int g_num_of_evq;
  #elif (defined(MWS_EVQ_CPP))
    extern int g_num_of_evq;
  #endif

  #if defined(MWS_INIT_CPP)
    // 現在存在的 evq.
    std::deque<mws_evq_id_t> g_alive_evq;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_EVQ_CPP))
    extern std::deque<mws_evq_id_t> g_alive_evq;
  #endif

  #if defined(MWS_INIT_CPP)
    // 管理 MWS 跨 thread 全域變數的鎖.
    pthread_mutex_t g_mws_global_mutex;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_EVQ_CPP) || \
         defined(MWS_RCV_CPP) || \
         defined(MWS_SRC_CPP) || \
         defined(MWS_UTIL_CPP))
    extern pthread_mutex_t g_mws_global_mutex;
  #endif

  #if defined(MWS_INIT_CPP)
    // 每個 fd (index)的詳細資料.
    // 注意: 1. 清除資料前要先確保 all_set 已經清除.
    //       2. 要在 close(fd) 前清除 g_fd_table 的資料.
    mws_fd_detail_t g_fd_table[MAX_FD_SIZE];
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_DEBUG_CPP) || \
         defined(MWS_EVQ_CPP) || \
         defined(MWS_SRC_CPP) || \
         defined(MWS_RCV_CPP) || \
         defined(MWS_UTIL_CPP))
    extern mws_fd_detail_t g_fd_table[];
  #endif

  #if defined(MWS_INIT_CPP)
    // mws_pkg_head_t 的 size.
    size_t g_size_of_mws_pkg_head;
  #elif (defined(MWS_EVQ_CPP) || \
         defined(MWS_SRC_CPP) || \
         defined(MWS_RCV_CPP))
    extern size_t g_size_of_mws_pkg_head;
  #endif

  #if defined(MWS_INIT_CPP)
    // mws_pkg_head_t 的 size.
    size_t g_size_of_mws_pkg_t;
  #elif (defined(MWS_EVQ_CPP) || \
         defined(MWS_UTIL_CPP))
    extern size_t g_size_of_mws_pkg_t;
  #endif

  #if defined(MWS_INIT_CPP)
    // mws_topic_name_msg_t 的 size.
    size_t g_size_of_mws_topic_name_msg_t;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_UTIL_CPP))
    extern size_t g_size_of_mws_topic_name_msg_t;
  #endif

  #if defined(MWS_INIT_CPP)
    // 轉換位元組順序的工具物件.
    mws_endianness_t g_endianness_obj;
  #elif (defined(MWS_EVQ_CPP) || \
         defined(MWS_RCV_CPP) || \
         defined(MWS_SRC_CPP))
    extern mws_endianness_t g_endianness_obj;
  #endif

  #if defined(MWS_INIT_CPP)
    // 現在時間(time_t 格式).
    time_t g_time_current;
    // 跨 thread 讀且全域變數 g_time_current 的鎖.
    pthread_mutex_t g_time_current_mutex;
  #elif (defined(MWS_CTX_CPP) || \
         defined(MWS_EVQ_CPP))
    extern time_t g_time_current;
    extern pthread_mutex_t g_time_current_mutex;
  #endif

  #if (MWS_DEBUG == 1)
    #if defined(MWS_DEBUG_CPP)
      std::deque<std::string> g_mws_debug_log;
      pthread_mutex_t g_mws_debug_log_mutex;
      pthread_t g_mws_debug_thread_id;
    #elif (defined(MWS_CTX_CPP) || \
           defined(MWS_EVQ_CPP) || \
           defined(MWS_INIT_CPP) || \
           defined(MWS_TYPE_DEFINITION_CPP))
      extern std::deque<std::string> g_mws_debug_log;
      extern pthread_mutex_t g_mws_debug_log_mutex;
      extern pthread_t g_mws_debug_thread_id;
    #endif
  #endif

} // namespace mws_global_variable

#endif /* MWS_GLOBAL_VARIABLE_H_ */
