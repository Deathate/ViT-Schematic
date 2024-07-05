//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_CTX_CPP 1

#include <algorithm>  // find().
#include <ctime>
#include <fcntl.h>
#include <arpa/inet.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <stdint.h>
#include <sched.h>  // sched_yield().
#include <sys/socket.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <netinet/tcp.h>
//#include <thread>
#include <unistd.h>

#include "../inc/mws_init.h"
#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_socket.h"
#include "../inc/mws_type_definition.h"
#include "../inc/mws_util.h"

void* ctx_thread_function(void* mws_ctx_ptr);

using namespace mws_global_variable;
using namespace mws_log;

mws_ctx_attr::mws_ctx_attr(std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  this->cfg_section = cfg_section;

  // ctx set from default.
  this->pthread_stack_size = 0;

  std::map<std::string, std::string> my_cfg;
  std::string default_section = "default_context_config_value";
  std::map<std::string, std::map<std::string, std::string> >::iterator it;
  it = g_config_mapping.find(default_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    std::string name("pthread_stack_size");
    this->pthread_stack_size = (size_t)atoll(my_cfg[name].c_str());
  }

  it = g_config_mapping.find(cfg_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    std::string name("pthread_stack_size");
    this->pthread_stack_size = (size_t)atoll(my_cfg[name].c_str());
  }

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return;
}

mws_ctx_attr::~mws_ctx_attr()
{
  return;
}

void mws_ctx_attr::mws_modify_ctx_attr(std::string attr_name,
                                       std::string attr_value)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  if (attr_name == "pthread_stack_size")
  {
    this->pthread_stack_size = (size_t)atoll(attr_value.c_str());
  }

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return;
}

int32_t mws_init_ctx(mws_ctx_t* ctx_ptr,
                     const bool is_from_cfg,
                     const mws_ctx_attr_t mws_ctx_attr,
                     const std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  ctx_ptr->object_status = 0;

  if (is_from_cfg == false)
  {
    ctx_ptr->cfg_section = mws_ctx_attr.cfg_section;
    ctx_ptr->pthread_stack_size = mws_ctx_attr.pthread_stack_size;
  } // if (is_from_cfg == false)
  else
  {
    // Begin: ctx set from default.
    ctx_ptr->pthread_stack_size = 0;
    // End: ctx set from default.
    // Begin: 從 cfg 的 default 取得設定值.
    std::map<std::string, std::string> my_cfg;
    std::string default_section = "default_context_config_value";
    std::map<std::string, std::map<std::string, std::string> >::iterator it;
    it = g_config_mapping.find(default_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定 pthread_stack_size.
      std::string name("pthread_stack_size");
      ctx_ptr->pthread_stack_size = (size_t)atoll(my_cfg[name].c_str());
    }
    // End: 從 cfg 的 default 取得設定值.
    // Begin: 從設定的 cfg section 取得設定值.
    it = g_config_mapping.find(cfg_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定 pthread_stack_size.
      std::string name("pthread_stack_size");
      ctx_ptr->pthread_stack_size = (size_t)atoll(my_cfg[name].c_str());
    }
    // End: 從設定的 cfg section 取得設定值.
  } // else of if (is_from_cfg == false)

  ctx_ptr->ctx_no = g_num_of_ctx++;

  // Begin: 初始化 ctx_list_owned_src_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_owned_src_mutex.

  // Begin: 初始化 ctx_list_owned_rcv_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_owned_rcv_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_owned_rcv_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_owned_rcv_mutex.

  // Begin: 初始化 ctx_list_owned_src_listen_fds_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_listen_fds_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_listen_fds_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_owned_src_listen_fds_mutex.

  // Begin: 初始化 ctx_list_owned_src_conn_fds_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_owned_src_conn_fds_mutex.

  // Begin: 初始化 ctx_list_owned_rcv_fds_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_owned_rcv_fds_mutex.

  // Begin: 初始化 ctx_list_wait_to_stop_src_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_wait_to_stop_src_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_wait_to_stop_src_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_wait_to_stop_src_mutex.

  // Begin: 初始化 ctx_list_wait_to_stop_rcv_mutex.
  {
    int rtv = pthread_mutex_init(&(ctx_ptr->ctx_list_wait_to_stop_rcv_mutex), NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(&(ctx_ptr->ctx_list_wait_to_stop_rcv_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_MUTEX_INIT;

      return -1;
    }
  }
  // End: 初始化 ctx_list_wait_to_stop_rcv_mutex.

  // 建立屬於此 ctx 的 timer_callback 工具物件.
  ctx_ptr->timer_callback_ptr = new mws_timer_callback_t(false);

  // ctx select thread 是否在運作中.
  // 等下要 create ctx thread 先把 must_stop_running_ctx_thread 設為 false.
  ctx_ptr->must_stop_running_ctx_thread = false;
  ctx_ptr->is_ctx_thread_running = false;

  // 初始化 all_set, rset, max_fd.
  FD_ZERO(&(ctx_ptr->all_set));
  FD_ZERO(&(ctx_ptr->rset));
  ctx_ptr->max_fd = -1;
  // 清空各 list.
  ctx_ptr->ctx_list_wait_to_connect_rcv_session.clear();
  ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.clear();
  ctx_ptr->ctx_list_wait_to_close_src_listen_fds.clear();
  ctx_ptr->ctx_list_wait_to_close_src_conn_fds.clear();
  ctx_ptr->ctx_list_wait_to_close_rcv_fds.clear();
  ctx_ptr->ctx_list_owned_src_listen_fds.clear();
  ctx_ptr->ctx_list_owned_src_conn_fds.clear();
  ctx_ptr->ctx_list_owned_rcv_fds.clear();

  // *** begin: create ctx thread.
  // 1. 設定 ctx thread 的屬性.
  pthread_attr_t attr;
  int pthread_rtv = 0;

  pthread_rtv = pthread_attr_init(&attr);
  if (pthread_rtv != 0)
  {
    std::string log_body;
    log_body = "pthread_attr_init() failed (rtv: " +
               std::to_string(pthread_rtv) +
               ", errno: " + std::to_string(errno) +
               ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

    //pthread_mutex_unlock(&g_mws_global_mutex);
    #if (MWS_DEBUG == 1)
      g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_mws_global_mutex_unlock();
    #endif

    return -1;
  }

  if (ctx_ptr->pthread_stack_size > 0)
  {
    pthread_rtv = pthread_attr_setstacksize(&attr, ctx_ptr->pthread_stack_size);
    if (pthread_rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_attr_setstacksize() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return -1;
    }
  }

  size_t curr_pthread_stack_size = 0;
  pthread_rtv = pthread_attr_getstacksize(&attr, &curr_pthread_stack_size);
  if (pthread_rtv != 0)
  {
    std::string log_body;
    log_body = "pthread_attr_getstacksize() failed (rtv: " +
               std::to_string(pthread_rtv) +
               ", errno: " + std::to_string(errno) +
               ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

    //pthread_mutex_unlock(&g_mws_global_mutex);
    #if (MWS_DEBUG == 1)
      g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_mws_global_mutex_unlock();
    #endif

    return -1;
  }
  // 若 ctx stack size 設定值小於 MIN_MWS_CTX_STACK_SIZE,
  // 將 ctx stack size 設定為 MIN_MWS_CTX_STACK_SIZE.
  if (curr_pthread_stack_size < MIN_MWS_CTX_STACK_SIZE)
  {
    pthread_rtv = pthread_attr_setstacksize(&attr, MIN_MWS_CTX_STACK_SIZE);
    if (pthread_rtv == 0)
    {
      std::string log_body;
      log_body = "pthread_stack_size(" +
                 std::to_string(curr_pthread_stack_size) +
                 ") is not enough and is changed to MIN_MWS_CTX_STACK_SIZE(" +
                 std::to_string(MIN_MWS_CTX_STACK_SIZE) + ")";
      write_to_log("", 1, "W", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "pthread_attr_setstacksize() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return -1;
    }
  }

  // 2. 建立 ctx thread.
  pthread_rtv = pthread_create(&(ctx_ptr->ctx_thread_id),
                               &attr,
                               ctx_thread_function,
                               (void*)ctx_ptr);
  if (pthread_rtv != 0)
  {
    std::string log_body;
    log_body = "pthread_create() failed (rtv: " +
               std::to_string(pthread_rtv) +
               ", errno: " + std::to_string(errno) +
               ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

    //pthread_mutex_unlock(&g_mws_global_mutex);
    #if (MWS_DEBUG == 1)
      g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_mws_global_mutex_unlock();
    #endif

    return -1;
  }

  // 3. 消滅 thread 屬性物件.
  pthread_rtv = pthread_attr_destroy(&attr);
  if (pthread_rtv != 0)
  {
    std::string log_body;
    log_body = "pthread_attr_destroy() failed (rtv: " +
               std::to_string(pthread_rtv) +
               ", errno: " + std::to_string(errno) +
               ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    ctx_ptr->object_status = MWS_ERROR_PTHREAD_CREATE;

    //pthread_mutex_unlock(&g_mws_global_mutex);
    #if (MWS_DEBUG == 1)
      g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_mws_global_mutex_unlock();
    #endif

    return -1;
  }
  // *** end: create ctx thread.

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return 0;
}

mws_ctx::mws_ctx(mws_ctx_attr_t mws_ctx_attr)
{
  // 無用但需要的變數.
  std::string cfg_section("");
  int32_t rtv = mws_init_ctx(this,
                             false,
                             mws_ctx_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_ctx constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_ctx constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_ctx::mws_ctx(std::string cfg_section)
{
  // 無用但需要的變數.
  mws_ctx_attr_t mws_ctx_attr("");
  int32_t rtv = mws_init_ctx(this,
                             true,
                             mws_ctx_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_ctx constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_ctx constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_ctx::~mws_ctx()
{
  //pthread_mutex_lock(&(this->ctx_list_owned_src_listen_fds_mutex));
  //pthread_mutex_lock(&(this->ctx_list_owned_src_conn_fds_mutex));
  //pthread_mutex_lock(&(this->ctx_list_owned_rcv_fds_mutex));
  //pthread_mutex_lock(&(this->ctx_list_wait_to_stop_src_mutex));
  //pthread_mutex_lock(&(this->ctx_list_wait_to_stop_rcv_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_wait_to_stop_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_wait_to_stop_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_src_listen_fds_mutex_lock();
    this->ctx_list_owned_src_conn_fds_mutex_lock();
    this->ctx_list_owned_rcv_fds_mutex_lock();
    this->ctx_list_wait_to_stop_src_mutex_lock();
    this->ctx_list_wait_to_stop_rcv_mutex_lock();
  #endif

  // 等待所有屬於此 ctx 的 fd 都結束工作才能開始解構此 ctx.
  while ((this->ctx_list_owned_src_listen_fds.size() != 0) ||
         (this->ctx_list_owned_src_conn_fds.size() != 0) ||
         (this->ctx_list_owned_rcv_fds.size() != 0) ||
         (this->ctx_list_wait_to_stop_src.size() != 0) ||
         (this->ctx_list_wait_to_stop_rcv.size() != 0))
  {
    //pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_rcv_mutex));
    //pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_src_mutex));
    //pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));
    //pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));
    //pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_list_wait_to_stop_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_wait_to_stop_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_list_wait_to_stop_rcv_mutex_unlock();
      this->ctx_list_wait_to_stop_src_mutex_unlock();
      this->ctx_list_owned_rcv_fds_mutex_unlock();
      this->ctx_list_owned_src_conn_fds_mutex_unlock();
      this->ctx_list_owned_src_listen_fds_mutex_unlock();
    #endif

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << "fd of ctx != 0" << std::endl;
    sleep(1);

    //pthread_mutex_lock(&(this->ctx_list_owned_src_listen_fds_mutex));
    //pthread_mutex_lock(&(this->ctx_list_owned_src_conn_fds_mutex));
    //pthread_mutex_lock(&(this->ctx_list_owned_rcv_fds_mutex));
    //pthread_mutex_lock(&(this->ctx_list_wait_to_stop_src_mutex));
    //pthread_mutex_lock(&(this->ctx_list_wait_to_stop_rcv_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_wait_to_stop_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      this->ctx_list_wait_to_stop_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_list_owned_src_listen_fds_mutex_lock();
      this->ctx_list_owned_src_conn_fds_mutex_lock();
      this->ctx_list_owned_rcv_fds_mutex_lock();
      this->ctx_list_wait_to_stop_src_mutex_lock();
      this->ctx_list_wait_to_stop_rcv_mutex_lock();
    #endif
  }
  //pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_rcv_mutex));
  //pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_src_mutex));
  //pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));
  //pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));
  //pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_wait_to_stop_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_wait_to_stop_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    this->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_wait_to_stop_rcv_mutex_unlock();
    this->ctx_list_wait_to_stop_src_mutex_unlock();
    this->ctx_list_owned_rcv_fds_mutex_unlock();
    this->ctx_list_owned_src_conn_fds_mutex_unlock();
    this->ctx_list_owned_src_listen_fds_mutex_unlock();
  #endif

  // Begin: 停止 ctx thread.
  {
    this->must_stop_running_ctx_thread = true;

    while (this->is_ctx_thread_running == true)
    {
      usleep(1000);
    }

    int rtv = pthread_join(this->ctx_thread_id, NULL);
    if (rtv != 0)
    {
      std::string log_body = "pthread_join() failed (rtv: " +
                             std::to_string(rtv) +
                             ", errno: " + std::to_string(errno) +
                             ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 停止 ctx thread.

  // 消滅 timer_callback 工具物件.
  delete this->timer_callback_ptr;
  this->timer_callback_ptr = NULL;

  std::string log_body = "mws_ctx destructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

void mws_ctx::update_max_fd(const fd_t fd)
{
  if (fd > this->max_fd)
  {
    this->max_fd = fd;
  }

  return;
}

void mws_ctx::clear_data_of_specified_rcv_from_ctx_list_wait_to_connect_rcv_session(const mws_rcv_t* rcv_ptr)
{
  std::deque<wait_to_connect_rcv_session_t>::iterator it = this->ctx_list_wait_to_connect_rcv_session.begin();
  while (it != this->ctx_list_wait_to_connect_rcv_session.end())
  {
    if (it->rcv_ptr == rcv_ptr)
    {
      it = this->ctx_list_wait_to_connect_rcv_session.erase(it);
    }
    else
    {
      if (it != this->ctx_list_wait_to_connect_rcv_session.end())
      {
        ++it;
      }
    }
  }

  return;
}

int mws_ctx::erase_ctx_list_wait_to_check_topic_rcv_session(const fd_t fd)
{
  std::deque<wait_to_check_topic_rcv_session_t>::iterator it = this->ctx_list_wait_to_check_topic_rcv_session.begin();
  while (it != this->ctx_list_wait_to_check_topic_rcv_session.end())
  {
    if (it->fd == fd)
    {
      this->ctx_list_wait_to_check_topic_rcv_session.erase(it);
      return 0;
    }
    else
    {
      if (it != this->ctx_list_wait_to_check_topic_rcv_session.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_ctx::erase_ctx_list_wait_to_check_topic_src_conn_session(const fd_t fd)
{
  std::deque<wait_to_check_topic_src_conn_session_t>::iterator it = this->ctx_list_wait_to_check_topic_src_conn_session.begin();
  while (it != this->ctx_list_wait_to_check_topic_src_conn_session.end())
  {
    if (it->fd == fd)
    {
      this->ctx_list_wait_to_check_topic_src_conn_session.erase(it);
      return 0;
    }
    else
    {
      if (it != this->ctx_list_wait_to_check_topic_src_conn_session.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_ctx::erase_ctx_list_wait_to_close_src_listen_fds(const fd_t fd)
{
  std::deque<fd_t>::iterator it = this->ctx_list_wait_to_close_src_listen_fds.begin();
  while (it != this->ctx_list_wait_to_close_src_listen_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_wait_to_close_src_listen_fds.erase(it);
      return 0;
    }
    else
    {
      if (it != this->ctx_list_wait_to_close_src_listen_fds.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_ctx::erase_ctx_list_wait_to_close_src_conn_fds(const fd_t fd)
{
  std::deque<fd_t>::iterator it = this->ctx_list_wait_to_close_src_conn_fds.begin();
  while (it != this->ctx_list_wait_to_close_src_conn_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_wait_to_close_src_conn_fds.erase(it);
      return 0;
    }
    else
    {
      if (it != this->ctx_list_wait_to_close_src_conn_fds.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_ctx::erase_ctx_list_wait_to_close_rcv_fds(const fd_t fd)
{
  std::deque<fd_t>::iterator it = this->ctx_list_wait_to_close_rcv_fds.begin();
  while (it != this->ctx_list_wait_to_close_rcv_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_wait_to_close_rcv_fds.erase(it);
      return 0;
    }
    else
    {
      if (it != this->ctx_list_wait_to_close_rcv_fds.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_ctx::erase_ctx_list_owned_src_listen_fds(const fd_t fd)
{
  //pthread_mutex_lock(&(this->ctx_list_owned_src_listen_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_src_listen_fds_mutex_lock();
  #endif

  std::deque<fd_t>::iterator it = this->ctx_list_owned_src_listen_fds.begin();
  while (it != this->ctx_list_owned_src_listen_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_owned_src_listen_fds.erase(it);

      //pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));
      #if (MWS_DEBUG == 1)
        this->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->ctx_list_owned_src_listen_fds_mutex_unlock();
      #endif

      return 0;
    }
    else
    {
      if (it != this->ctx_list_owned_src_listen_fds.end())
      {
        ++it;
      }
    }
  }

  //pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_src_listen_fds_mutex_unlock();
  #endif

  return 1;
}

int mws_ctx::erase_ctx_list_owned_src_conn_fds(const fd_t fd)
{
  //pthread_mutex_lock(&(this->ctx_list_owned_src_conn_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_src_conn_fds_mutex_lock();
  #endif

  std::deque<fd_t>::iterator it = this->ctx_list_owned_src_conn_fds.begin();
  while (it != this->ctx_list_owned_src_conn_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_owned_src_conn_fds.erase(it);

      //pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));
      #if (MWS_DEBUG == 1)
        this->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->ctx_list_owned_src_conn_fds_mutex_unlock();
      #endif

      return 0;
    }
    else
    {
      if (it != this->ctx_list_owned_src_conn_fds.end())
      {
        ++it;
      }
    }
  }

  //pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_src_conn_fds_mutex_unlock();
  #endif

  return 1;
}

int mws_ctx::erase_ctx_list_owned_rcv_fds(const fd_t fd)
{
  //pthread_mutex_lock(&(this->ctx_list_owned_rcv_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_rcv_fds_mutex_lock();
  #endif

  std::deque<fd_t>::iterator it = this->ctx_list_owned_rcv_fds.begin();
  while (it != this->ctx_list_owned_rcv_fds.end())
  {
    if (*it == fd)
    {
      this->ctx_list_owned_rcv_fds.erase(it);
      //pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));
      #if (MWS_DEBUG == 1)
        this->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->ctx_list_owned_rcv_fds_mutex_unlock();
      #endif

      return 0;
    }
    else
    {
      if (it != this->ctx_list_owned_rcv_fds.end())
      {
        ++it;
      }
    }
  }
  //pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_list_owned_rcv_fds_mutex_unlock();
  #endif

  return 1;
}

std::string mws_ctx::mws_get_cfg_section()
{
  return this->cfg_section;
}

uint32_t mws_ctx::mws_get_object_status()
{
  return this->object_status;
}

int32_t mws_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
                                    timer_callback_t cb_function,
                                    void* custom_data_ptr,
                                    long delay_usec,
                                    bool is_recurring)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->schedule_timer(cb_function,
                                                      custom_data_ptr,
                                                      delay_usec,
                                                      is_recurring);
  }
  else
  {
    return this->timer_callback_ptr->schedule_timer(cb_function,
                                                    custom_data_ptr,
                                                    delay_usec,
                                                    is_recurring);
  }
}

int32_t mws_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
                                    timer_callback_t cb_function,
                                    void* custom_data_ptr,
                                    long delay_sec,
                                    long delay_usec,
                                    bool is_recurring)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->schedule_timer(cb_function,
                                                       custom_data_ptr,
                                                       delay_sec,
                                                       delay_usec,
                                                       is_recurring);
  }
  else
  {
    return this->timer_callback_ptr->schedule_timer(cb_function,
                                                    custom_data_ptr,
                                                    delay_sec,
                                                    delay_usec,
                                                    is_recurring);
  }
}

int32_t mws_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
                                    timer_callback_t cb_function,
                                    void* custom_data_ptr,
                                    tmvl_t time_tv)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->schedule_timer(cb_function,
                                                      custom_data_ptr,
                                                      time_tv);
  }
  else
  {
    return this->timer_callback_ptr->schedule_timer(cb_function,
                                                    custom_data_ptr,
                                                    time_tv);
  }
}

int32_t mws_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
                                    timer_callback_t cb_function,
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
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->schedule_timer(cb_function,
                                                      custom_data_ptr,
                                                      year,
                                                      mon,
                                                      mday,
                                                      hour,
                                                      min,
                                                      sec,
                                                      usec,
                                                      isdst);
  }
  else
  {
    return this->timer_callback_ptr->schedule_timer(cb_function,
                                                    custom_data_ptr,
                                                    year,
                                                    mon,
                                                    mday,
                                                    hour,
                                                    min,
                                                    sec,
                                                    usec,
                                                    isdst);
  }
}

int32_t mws_ctx::mws_cancel_timer(mws_evq_t* evq_ptr,
                                  const int32_t timer_id)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->cancel_timer(timer_id);
  }
  else
  {
    return this->timer_callback_ptr->cancel_timer(timer_id);
  }
}

std::string mws_ctx::mws_timer_version(mws_evq_t* evq_ptr)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->version();
  }
  else
  {
    return this->timer_callback_ptr->version();
  }
}

int32_t mws_ctx::mws_show_all_timer_detail(mws_evq_t* evq_ptr)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->show_all_timer_detail();
  }
  else
  {
    return this->timer_callback_ptr->show_all_timer_detail();
  }
}

int32_t mws_ctx::mws_show_num_of_timer_with_evq(mws_evq_t* evq_ptr)
{
  if (evq_ptr != NULL)
  {
    return evq_ptr->timer_callback_ptr->show_num_of_timer();
  }

  return 0;
}

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_owned_src_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_owned_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_owned_src_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_owned_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_owned_src_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_owned_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_owned_src_mutex));
  }

  void mws_ctx::ctx_list_owned_src_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_owned_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_owned_src_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_owned_src_mutex));
    return;
  }

  int mws_ctx::ctx_list_owned_src_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_owned_src_mutex));
  }

  void mws_ctx::ctx_list_owned_src_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_owned_rcv_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_owned_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_owned_rcv_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_owned_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_owned_rcv_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_owned_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_owned_rcv_mutex));
  }

  void mws_ctx::ctx_list_owned_rcv_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_rcv_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_owned_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_owned_rcv_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_owned_rcv_mutex));
    return;
  }

  int mws_ctx::ctx_list_owned_rcv_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_owned_rcv_mutex));
  }

  void mws_ctx::ctx_list_owned_rcv_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_rcv_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_owned_src_listen_fds_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_owned_src_listen_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_owned_src_listen_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_owned_src_listen_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_owned_src_listen_fds_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_owned_src_listen_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_owned_src_listen_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_src_listen_fds_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_owned_src_listen_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_owned_src_listen_fds_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_owned_src_listen_fds_mutex));
    return;
  }

  int mws_ctx::ctx_list_owned_src_listen_fds_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_owned_src_listen_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_src_listen_fds_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_listen_fds_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_owned_src_conn_fds_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_owned_src_conn_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_owned_src_conn_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_owned_src_conn_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_owned_src_conn_fds_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_owned_src_conn_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_owned_src_conn_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_src_conn_fds_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_owned_src_conn_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_owned_src_conn_fds_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_owned_src_conn_fds_mutex));
    return;
  }

  int mws_ctx::ctx_list_owned_src_conn_fds_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_owned_src_conn_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_src_conn_fds_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_src_conn_fds_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_owned_rcv_fds_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_owned_rcv_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_owned_rcv_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_owned_rcv_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_owned_rcv_fds_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_owned_rcv_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_owned_rcv_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_rcv_fds_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_owned_rcv_fds_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_owned_rcv_fds_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_owned_rcv_fds_mutex));
    return;
  }

  int mws_ctx::ctx_list_owned_rcv_fds_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_owned_rcv_fds_mutex));
  }

  void mws_ctx::ctx_list_owned_rcv_fds_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_owned_rcv_fds_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_wait_to_stop_src_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_wait_to_stop_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_wait_to_stop_src_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_wait_to_stop_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_wait_to_stop_src_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_wait_to_stop_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_wait_to_stop_src_mutex));
  }

  void mws_ctx::ctx_list_wait_to_stop_src_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_src_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_wait_to_stop_src_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_wait_to_stop_src_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_wait_to_stop_src_mutex));
    return;
  }

  int mws_ctx::ctx_list_wait_to_stop_src_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_wait_to_stop_src_mutex));
  }

  void mws_ctx::ctx_list_wait_to_stop_src_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_src_mutex));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_ctx::ctx_list_wait_to_stop_rcv_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock ctx_list_wait_to_stop_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->ctx_list_wait_to_stop_rcv_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking ctx_list_wait_to_stop_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_ctx::ctx_list_wait_to_stop_rcv_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock ctx_list_wait_to_stop_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->ctx_list_wait_to_stop_rcv_mutex));
  }

  void mws_ctx::ctx_list_wait_to_stop_rcv_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_rcv_mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock ctx_list_wait_to_stop_rcv_mutex ctx_no:";
      log += std::to_string(this->ctx_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_ctx::ctx_list_wait_to_stop_rcv_mutex_lock()
  {
    pthread_mutex_lock(&(this->ctx_list_wait_to_stop_rcv_mutex));
    return;
  }

  int mws_ctx::ctx_list_wait_to_stop_rcv_mutex_trylock()
  {
    return pthread_mutex_trylock(&(this->ctx_list_wait_to_stop_rcv_mutex));
  }

  void mws_ctx::ctx_list_wait_to_stop_rcv_mutex_unlock()
  {
    pthread_mutex_unlock(&(this->ctx_list_wait_to_stop_rcv_mutex));
    return;
  }
#endif

void* ctx_thread_function(void* mws_ctx_ptr)
{
  //std::thread::id tid = std::this_thread::get_id();
  //std::cout << "ctx thread id : " << tid << std::endl;

  std::string log_body;

  mws_ctx_t* ctx_ptr = (mws_ctx_t*)mws_ctx_ptr;

  timeval_t select_timeout;

  // interval_heartbeat_sec: 每間隔 interval_heartbeat_sec 秒, 每個 src/rcv 的 session 傳送一個 heartbeat message.
  const time_t interval_heartbeat_sec = 3;
  // interval_batch_job_sec: 每間隔 interval_batch_job_sec 秒, 執行一次批次作業.
  const time_t interval_batch_job_sec = 5;
  // interval_reconnect_sec: rcv 在 connect 超過十次失敗後, 每間隔 interval_reconnect_sec 秒, 重新 connect 一次.
  const time_t interval_reconnect_sec = 30;

  time_t t_prev_heartbeat = time(NULL);
  time_t t_prev_batch_job = time(NULL);
  time_t t_prev_reconnect = time(NULL);

  pthread_mutex_lock(&(g_time_current_mutex));
  g_time_current = time(NULL);
  time_t t_current = g_time_current;
  pthread_mutex_unlock(&(g_time_current_mutex));

  ctx_ptr->is_ctx_thread_running = true;

  while (ctx_ptr->must_stop_running_ctx_thread == false)
  {
    // timer callback 作業.
    ctx_ptr->timer_callback_ptr->timer_manager();

    // 取得現在時間.
    pthread_mutex_lock(&(g_time_current_mutex));
    g_time_current = time(NULL);
    t_current = g_time_current;
    pthread_mutex_unlock(&(g_time_current_mutex));

    // Begin: 每間隔 interval_heartbeat_sec 秒, 每個 src/rcv 的 session 傳送一個 heartbeat message.
    if ((t_current - t_prev_heartbeat) > interval_heartbeat_sec)
    {
      t_prev_heartbeat = t_current;

      // Begin: 依照 ctx_list_owned_src 傳送 heartbeat.
      {
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_src_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_src_mutex_lock();
        #endif

        std::deque<mws_src_t*>::iterator it = ctx_ptr->ctx_list_owned_src.begin();
        while (it != ctx_ptr->ctx_list_owned_src.end())
        {
          // 傳送 heartbeat.
          (*it)->mws_src_send_heartbeat();

          //std::cout << __func__ << ":" << __LINE__ << " src send hb" << std::endl;

          if (it != ctx_ptr->ctx_list_owned_src.end())
          {
            ++it;
          }
        }
        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_src_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_src_mutex_unlock();
        #endif
      }
      // End: 依照 ctx_list_owned_src 傳送 heartbeat.

      // Begin: 依照 ctx_list_owned_rcv 傳送 heartbeat.
      {
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_rcv_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_rcv_mutex_lock();
        #endif

        std::deque<mws_rcv_t*>::iterator it = ctx_ptr->ctx_list_owned_rcv.begin();
        while (it != ctx_ptr->ctx_list_owned_rcv.end())
        {
          // 傳送 heartbeat.
          (*it)->mws_rcv_send_heartbeat();

          //std::cout << __func__ << ":" << __LINE__ << " rcv send hb" << std::endl;

          if (it != ctx_ptr->ctx_list_owned_rcv.end())
          {
            ++it;
          }
        }
        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_rcv_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_rcv_mutex_unlock();
        #endif
      }
      // End: 依照 ctx_list_owned_rcv 傳送 heartbeat.
    }
    // End: 每間隔 interval_heartbeat_sec 秒, 每個 src/rcv 的 session 傳送一個 heartbeat message.

    // Begin: 每間隔 t_prev_batch_job 秒, 批次執行的工作的區段.
    // 目前作業項目:
    //   1. 依照 ctx_list_owned_src_conn_fds 檢查 fd 是否 timed out.
    //   2. 依照 ctx_list_owned_rcv_fds 檢查 fd 是否 timed out.
    //   3. 依照 ctx_list_wait_to_stop_src 內容 停止 src 機能.
    //   4. 依照 ctx_list_wait_to_stop_rcv 內容 停止 rcv 機能.
    //   5. 依照 ctx_list_wait_to_close_src_conn_fds 內容 close fd.
    //   6. 依照 ctx_list_wait_to_close_rcv_fds 內容 close fd.
    //   7. 依照 ctx_list_wait_to_check_topic_src_conn_session 內容中的 status 執行發送 0xFE 到 rcv.
    //   8. 依照 ctx_list_wait_to_check_topic_rcv_session 內容中的 status 執行發送 0xFF 到 src conn.
    //   9. 依照 ctx_list_wait_to_connect_rcv_session 內容執行 rcv 連線到 src.
    // 希望把間隔時間控制在 t_prev_batch_job 秒以上.
    if ((t_current - t_prev_batch_job) > interval_batch_job_sec)
    {
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

      // 將 t_prev_batch_job 更新為現在時間.
      t_prev_batch_job = t_current;

      // 是否該做 reconnect.
      bool flag_time_to_reconnect = false;
      // 如果現在時間已經距上次做 reconnect 超過 interval_reconnect_sec,
      // 將 flag_time_to_reconnect 設為 true,
      // 並將 t_prev_reconnect 更新為現在時間.
      if ((t_current - t_prev_reconnect) > interval_reconnect_sec)
      {
        flag_time_to_reconnect = true;
        t_prev_reconnect = t_current;
      }

      //std::cout << "time:" << std::to_string(t_current) << std::endl;

      // Begin: 1. 依照 ctx_list_owned_src_conn_fds 檢查 fd 是否 timed out.
      {
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock();
        #endif

        std::deque<fd_t> temp_fds = ctx_ptr->ctx_list_owned_src_conn_fds;

        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock();
        #endif

        std::deque<fd_t>::iterator it = temp_fds.begin();
        while (it != temp_fds.end())
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif

          if ((g_fd_table[*it].status == FD_STATUS_SRC_CONN_READY) &&
              (g_fd_table[*it].msg_evq_ptr->empty(g_fd_table[*it].msg_evq_number) == true))
          {
            if ((t_current - g_fd_table[*it].last_heartbeat_time) > SESSION_TIMED_OUT_SEC)
            {
              {
                std::string log_body = "t_current: " + std::to_string(t_current) +
                                       ", last_heartbeat_time" + std::to_string(g_fd_table[*it].last_heartbeat_time) +
                                       ", SESSION_TIMED_OUT_SEC:" + std::to_string(SESSION_TIMED_OUT_SEC);
                write_to_log(g_fd_table[*it].src_conn_ptr->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }

              std::string log_body = "session of src conn fd: " + std::to_string(*it) + " is timed out.";
              write_to_log(g_fd_table[*it].src_conn_ptr->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
              g_fd_table[*it].src_conn_ptr->src_send_error(*it, __func__, __LINE__);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

              //std::cout << __func__ << ":" << __LINE__ << " src conn timed out" << std::endl;
            }
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_unlock();
          #endif

          if (it != temp_fds.end())
          {
            ++it;
          }
        }
      }
      // End: 1. 依照 ctx_list_owned_src_conn_fds 檢查 fd 是否 timed out.

      // Begin: 2. 依照 ctx_list_owned_rcv_fds 檢查 fd 是否 timed out.
      {
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock();
        #endif

        std::deque<fd_t> temp_fds = ctx_ptr->ctx_list_owned_rcv_fds;

        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock();
        #endif

        std::deque<fd_t>::iterator it = temp_fds.begin();
        while (it != temp_fds.end())
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif

          if ((g_fd_table[*it].status == FD_STATUS_RCV_READY) &&
              (g_fd_table[*it].msg_evq_ptr->empty(g_fd_table[*it].msg_evq_number) == true))
          {
            if ((t_current - g_fd_table[*it].last_heartbeat_time) > SESSION_TIMED_OUT_SEC)
            {
              {
                std::string log_body = "t_current: " + std::to_string(t_current) +
                                       ", last_heartbeat_time: " + std::to_string(g_fd_table[*it].last_heartbeat_time) +
                                       ", SESSION_TIMED_OUT_SEC:" + std::to_string(SESSION_TIMED_OUT_SEC);
                write_to_log(g_fd_table[*it].rcv_ptr->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }

              std::string log_body = "session of rcv fd: " + std::to_string(*it) + " is timed out.";
              write_to_log(g_fd_table[*it].rcv_ptr->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

              g_fd_table[*it].rcv_ptr->rcv_send_error(*it, __func__, __LINE__);

              //std::cout << __func__ << ":" << __LINE__ << " rcv conn timed out" << std::endl;
            }
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_unlock();
          #endif

          if (it != temp_fds.end())
          {
            ++it;
          }
        }
      }
      // End: 2. 依照 ctx_list_owned_rcv_fds 檢查 fd 是否 timed out.

      // Begin: 3. 依照 ctx_list_wait_to_stop_src 內容 停止 src 機能.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_ptr->ctx_list_wait_to_stop_src.size():" << std::to_string(ctx_ptr->ctx_list_wait_to_stop_src.size()) << std::endl;
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_wait_to_stop_src_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock();
        #endif

        //std::cout << __func__ << ":" << __LINE__ << " " << std::endl;
        std::deque<mws_src_t*>::iterator it = ctx_ptr->ctx_list_wait_to_stop_src.begin();
        //std::cout << __func__ << ":" << __LINE__ << " " << std::endl;

        while (it != ctx_ptr->ctx_list_wait_to_stop_src.end())
        {
          //std::cout << __func__ << ":" << __LINE__ << " " << std::endl;

          mws_src_t* src_ptr = *it;

          std::string topic_name = src_ptr->topic_name;

          #if (MWS_DEBUG == 1)
            src_ptr->evq_ptr->evq_cond_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            src_ptr->evq_ptr->evq_cond_lock();
          #endif

          pthread_cond_signal(&(src_ptr->evq_ptr->cond_select_done));

          #if (MWS_DEBUG == 1)
            src_ptr->evq_ptr->evq_cond_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            src_ptr->evq_ptr->evq_cond_unlock();
          #endif

          #if (MWS_DEBUG == 1)
            src_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            src_ptr->evq_ptr->evq_lock();
          #endif

          // Begin: 走過 src_connect_fds.
          size_t num_of_src_conn_fd = src_ptr->src_connect_fds.size();
          for (size_t i = 0; i < num_of_src_conn_fd; ++i)
          {
            //std::cout << __func__ << ":" << __LINE__ << " " << std::endl;

            fd_t fd = src_ptr->src_connect_fds[i];

            #if (MWS_DEBUG == 1)
              g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[fd].fd_lock();
            #endif

            //  1. 清除 mws_ctx::all_set 內相同 fd 的資料.
            {
              FD_CLR(fd, &ctx_ptr->all_set);

              std::string log_body = "Remove src connect fd: " +
                                     std::to_string(fd) +
                                     " from all_set ";
              write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
            }
            //  2. 清除 mws_ctx::rset 內相同 fd 的資料.
            {
              FD_CLR(fd, &ctx_ptr->rset);

              std::string log_body = "Remove src connect fd: " +
                                     std::to_string(fd) +
                                     " from rset ";
              write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
            }
            //  3. 清除 mws_ctx::ctx_list_wait_to_check_topic_src_conn_session 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_wait_to_check_topic_src_conn_session(fd);
              if (rtv != 0)
              {
                // ctx_list_wait_to_check_topic_src_conn_session 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_wait_to_check_topic_src_conn_session";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  4. 清除 mws_ctx::ctx_list_wait_to_close_src_conn_fds 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_wait_to_close_src_conn_fds(fd);
              if (rtv != 0)
              {
                // ctx_list_wait_to_close_src_conn_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_wait_to_close_src_conn_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  5. 清除 mws_ctx::ctx_list_owned_src_conn_fds 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_owned_src_conn_fds(fd);
              if (rtv != 0)
              {
                // ctx_list_owned_src_conn_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_owned_src_conn_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  6. 清除 mws_evq::connect_event_queue 相同 fd 的資料.
            {
              while (src_ptr->evq_ptr->connect_event_queue.empty() == false)
              {
                mws_event_t* event_ptr = src_ptr->evq_ptr->connect_event_queue.front();

                if (event_ptr->fd == fd)
                {
                  // 刪除 event 佔用的記憶體空間.
                  delete event_ptr;
                  // 把刪除的 event 從 connect_event_queue 中 pop 掉.
                  src_ptr->evq_ptr->connect_event_queue.pop();
                }
              }
            }
            //  7. 清除 mws_evq::disconnect_event_queue 相同 fd 的資料.
            {
              while (src_ptr->evq_ptr->disconnect_event_queue.empty() == false)
              {
                mws_event_t* event_ptr = src_ptr->evq_ptr->disconnect_event_queue.front();

                if (event_ptr->fd == fd)
                {
                  int rtv = (*(g_fd_table[fd].src_conn_ptr->cb_ptr))(event_ptr,
                                                                     g_fd_table[fd].src_conn_ptr->custom_data_ptr,
                                                                     g_fd_table[fd].src_conn_ptr->custom_data_size);
                  if (rtv != 0)
                  {
                    std::string log_body =
                      "call callback function for src(" +
                      g_fd_table[fd].src_conn_ptr->topic_name +
                      ", " + event_ptr->src_addr.str_ip + ":" +
                      event_ptr->src_addr.str_port +
                      ") failed (rtv: " + std::to_string(rtv) + ")";
                    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
                  }

                  // 刪除 event 佔用的記憶體空間.
                  delete event_ptr;
                  // 把處理過的 event 從 disconnect_event_queue 中 pop 掉.
                  src_ptr->evq_ptr->disconnect_event_queue.pop();
                }
              }
            }
            //  8. 清除 mws_evq::evq_list_owned_fds 相同 fd (src conn fd)的資料.
            {
              //std::cout << __func__ << ":" << __LINE__ << " evq_list_owned_fds.size():" << std::to_string(src_ptr->evq_ptr->evq_list_owned_fds.size()) << std::endl;
              int rtv = src_ptr->evq_ptr->erase_evq_list_owned_fds(fd);
              if (rtv != 0)
              {
                // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in evq_list_owned_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  9. 初始化 g_fd_table.
            {
              g_fd_table[fd].fd_init(false);
              //std::cout << __func__ << ":" << __LINE__ << " fd_init()-fd:" << fd << std::endl;
            }
            // 10. close fd.
            {
              mws_close(fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body = "close src conn fd: " + std::to_string(fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }

            #if (MWS_DEBUG == 1)
              g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[fd].fd_unlock();
            #endif
          }
          // End: 走過 src_connect_fds.

          // 清除整個 src_connect_fds.
          src_ptr->src_connect_fds.clear();

          #if (MWS_DEBUG == 1)
            g_fd_table[src_ptr->src_listen_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[src_ptr->src_listen_fd].fd_lock();
          #endif

          // 清除 mws_ctx::all_set 內相同 src_listen_fd 的資料.
          {
            FD_CLR(src_ptr->src_listen_fd, &ctx_ptr->all_set);

            std::string log_body = "Remove src listen fd: " +
                                   std::to_string(src_ptr->src_listen_fd) +
                                   " from all_set ";
            write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
          }

          // 清除 mws_ctx::rset 內相同 src_listen_fd 的資料.
          {
            FD_CLR(src_ptr->src_listen_fd, &ctx_ptr->rset);

            std::string log_body = "Remove src listen fd: " +
                                   std::to_string(src_ptr->src_listen_fd) +
                                   " from rset ";
            write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
          }

          // 清除 mws_ctx::ctx_list_wait_to_close_src_listen_fds 相同 src_listen_fd 的資料.
          {
            int rtv = ctx_ptr->erase_ctx_list_wait_to_close_src_listen_fds(src_ptr->src_listen_fd);
            if (rtv != 0)
            {
              // ctx_list_wait_to_close_src_listen_fds 沒有該 fd 資料, 刷錯誤訊息.
              std::string log_body;
              log_body = "fd: " + std::to_string(src_ptr->src_listen_fd) + " does not exist in ctx_list_wait_to_close_src_listen_fds";
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
          }

          // 清除 mws_ctx::ctx_list_owned_src_listen_fds 相同 src_listen_fd 的資料.
          {
            int rtv = ctx_ptr->erase_ctx_list_owned_src_listen_fds(src_ptr->src_listen_fd);
            if (rtv != 0)
            {
              // ctx_list_owned_src_listen_fds 沒有該 fd 資料, 刷錯誤訊息.
              std::string log_body;
              log_body = "fd: " + std::to_string(src_ptr->src_listen_fd) + " does not exist in ctx_list_owned_src_listen_fds";
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
          }

          // 初始化 src_listen_fd 的 g_fd_table.
          {
            g_fd_table[src_ptr->src_listen_fd].fd_init(false);
            //std::cout << __func__ << ":" << __LINE__ << " fd_init()-fd:" << src_ptr->src_listen_fd << std::endl;
          }

          // close src_listen_fd.
          {
            mws_close(src_ptr->src_listen_fd);
            if (g_mws_log_level >= 1)
            {
              std::string log_body = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
          }

          // 回收 custom_data_ptr 指向的空間.
          free(src_ptr->custom_data_ptr);

          #if (MWS_DEBUG == 1)
            g_fd_table[src_ptr->src_listen_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[src_ptr->src_listen_fd].fd_unlock();
          #endif

          #if (MWS_DEBUG == 1)
            src_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            src_ptr->evq_ptr->evq_unlock();
          #endif

          // 可以解構此 src.
          src_ptr->flag_ready_to_release_src = true;

          // 處理下一筆資料.
          ++it;
        }
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
        // 清除 ctx_list_wait_to_stop_src 的全部內容.
        ctx_ptr->ctx_list_wait_to_stop_src.clear();
        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_wait_to_stop_src_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock();
        #endif
      }
      // End: 3. 依照 ctx_list_wait_to_stop_src 內容 停止 src 機能.

      // Begin: 4. 依照 ctx_list_wait_to_stop_rcv 內容 停止 rcv 機能.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_stop_rcv" << std::endl;
        //pthread_mutex_lock(&(ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock();
        #endif

        std::deque<mws_rcv_t*>::iterator it = ctx_ptr->ctx_list_wait_to_stop_rcv.begin();
        while (it != ctx_ptr->ctx_list_wait_to_stop_rcv.end())
        {
          mws_rcv_t* rcv_ptr = *it;

          std::string topic_name = rcv_ptr->topic_name;

          #if (MWS_DEBUG == 1)
            rcv_ptr->evq_ptr->evq_cond_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            rcv_ptr->evq_ptr->evq_cond_lock();
          #endif

          pthread_cond_signal(&(rcv_ptr->evq_ptr->cond_select_done));

          #if (MWS_DEBUG == 1)
            rcv_ptr->evq_ptr->evq_cond_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            rcv_ptr->evq_ptr->evq_cond_unlock();
          #endif

          #if (MWS_DEBUG == 1)
            rcv_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            rcv_ptr->evq_ptr->evq_lock();
          #endif

          // Begin: 走過 rcv_connect_fds.
          size_t num_of_rcv_conn_fd = rcv_ptr->rcv_connect_fds.size();
          for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)
          {
            fd_t fd = rcv_ptr->rcv_connect_fds[i];
            #if (MWS_DEBUG == 1)
              g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[fd].fd_lock();
            #endif

            //  1. 清除 mws_ctx::all_set 內相同 fd 的資料.
            {
              FD_CLR(fd, &ctx_ptr->all_set);

              std::string log_body = "Remove rcv connect fd: " +
                                     std::to_string(fd) +
                                     " from all_set ";
              write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);

            }
            //  2. 清除 mws_ctx::rset 內相同 fd 的資料.
            {
              FD_CLR(fd, &ctx_ptr->rset);

              std::string log_body = "Remove rcv connect fd: " +
                                     std::to_string(fd) +
                                     " from rset ";
              write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
            }
            //  3. 清除 mws_ctx::ctx_list_wait_to_check_topic_rcv_session 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_wait_to_check_topic_rcv_session(fd);
              if (rtv != 0)
              {
                // ctx_list_wait_to_check_topic_rcv_session 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_wait_to_check_topic_rcv_session";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  4. 清除 mws_ctx::ctx_list_wait_to_close_rcv_fds 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_wait_to_close_rcv_fds(fd);
              if (rtv != 0)
              {
                // ctx_list_wait_to_close_rcv_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_wait_to_close_rcv_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  5. 清除 mws_ctx::ctx_list_owned_rcv_conn_fds 相同 fd 的資料.
            {
              int rtv = ctx_ptr->erase_ctx_list_owned_rcv_fds(fd);
              if (rtv != 0)
              {
                // ctx_list_owned_rcv_conn_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_owned_rcv_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  6. 清除 mws_evq::connect_event_queue 相同 fd 的資料.
            {
              while (rcv_ptr->evq_ptr->connect_event_queue.empty() == false)
              {
                mws_event_t* event_ptr = rcv_ptr->evq_ptr->connect_event_queue.front();

                if (event_ptr->fd == fd)
                {
                  // 刪除 event 佔用的記憶體空間.
                  delete event_ptr;
                  // 把刪除的 event 從 connect_event_queue 中 pop 掉.
                  rcv_ptr->evq_ptr->connect_event_queue.pop();
                }
              }
            }
            //  7. 清除 mws_evq::disconnect_event_queue 相同 fd 的資料.
            {
              while (rcv_ptr->evq_ptr->disconnect_event_queue.empty() == false)
              {
                mws_event_t* event_ptr = rcv_ptr->evq_ptr->disconnect_event_queue.front();

                if (event_ptr->fd == fd)
                {
                  int rtv = (*(g_fd_table[fd].rcv_ptr->cb_ptr))(event_ptr,
                                                                g_fd_table[fd].rcv_ptr->custom_data_ptr,
                                                                g_fd_table[fd].rcv_ptr->custom_data_size);
                  if (rtv != 0)
                  {
                    std::string log_body =
                      "call callback function for rcv(" +
                      g_fd_table[fd].rcv_ptr->topic_name +
                      ", " + event_ptr->rcv_addr.str_ip + ":" +
                      event_ptr->rcv_addr.str_port +
                      ") failed (rtv: " + std::to_string(rtv) + ")";
                    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
                  }

                  // 刪除 event 佔用的記憶體空間.
                  delete event_ptr;
                  // 把處理過的 event 從 disconnect_event_queue 中 pop 掉.
                  rcv_ptr->evq_ptr->disconnect_event_queue.pop();
                }
              }
            }
            //  8. 清除 mws_evq::evq_list_owned_fds 相同 fd (rcv fd)的資料.
            {
              int rtv = rcv_ptr->evq_ptr->erase_evq_list_owned_fds(fd);
              if (rtv != 0)
              {
                // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(fd) + " does not exist in evq_list_owned_fds";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            //  9. 初始化 g_fd_table.
            {
              g_fd_table[fd].fd_init(false);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " fd_init()-fd:" << fd << std::endl;
            }
            // 10. close fd.
            {
              mws_close(fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body = "close rcv fd: " + std::to_string(fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }

            #if (MWS_DEBUG == 1)
              g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[fd].fd_unlock();
            #endif
          }
          // End: 走過 rcv_connect_fds.

          // 清除 mws_ctx::ctx_list_wait_to_connect_rcv_session 屬於 rcv_ptr 的資料.
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_connect_rcv_session.size():" << std::to_string(ctx_ptr->ctx_list_wait_to_connect_rcv_session.size()) << std::endl;
          ctx_ptr->clear_data_of_specified_rcv_from_ctx_list_wait_to_connect_rcv_session(rcv_ptr);
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_connect_rcv_session.size():" << std::to_string(ctx_ptr->ctx_list_wait_to_connect_rcv_session.size()) << std::endl;
          // 清除整個 rcv_connect_fds.
          rcv_ptr->rcv_connect_fds.clear();
          // 回收 custom_data_ptr 指向的空間.
          free(rcv_ptr->custom_data_ptr);

          #if (MWS_DEBUG == 1)
            rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            rcv_ptr->evq_ptr->evq_unlock();
          #endif

          // 可以解構此 rcv.
          rcv_ptr->flag_ready_to_release_rcv = true;
          // 處理下一筆資料.
          ++it;
        }

        // 清除 ctx_list_wait_to_stop_rcv 的全部內容.
        ctx_ptr->ctx_list_wait_to_stop_rcv.clear();
        //pthread_mutex_unlock(&(ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
        #if (MWS_DEBUG == 1)
          ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock();
        #endif
      }
      // End: 4. 依照 ctx_list_wait_to_stop_rcv 內容 停止 rcv 機能.

      // Begin: 5. 依照 ctx_list_wait_to_close_src_conn_fds 內容 close fd.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_close_src_fds" << std::endl;

        std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_wait_to_close_src_conn_fds.begin();
        while (it != ctx_ptr->ctx_list_wait_to_close_src_conn_fds.end())
        {
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif

          // Begin: 將 g_fd_table 重新初始化.
          {
            g_fd_table[*it].fd_init(true);
            //std::cout << __func__ << ":" << __LINE__ << " fd_init()-fd:" << *it << std::endl;
          }
          // End: 將 g_fd_table 重新初始化.

          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
          // closd fd.
          mws_close(*it);
          if (g_mws_log_level >= 1)
          {
            std::string log_body = "close src conn fd: " + std::to_string(*it);
            write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_unlock();
          #endif

          // 處理下一筆資料.
          ++it;
        }
        // 清除 ctx_list_wait_to_close_src_conn_fds 的全部內容.
        ctx_ptr->ctx_list_wait_to_close_src_conn_fds.clear();
      }
      // End: 5. 依照 ctx_list_wait_to_close_src_conn_fds 內容 close fd.

      // Begin: 6. 依照 ctx_list_wait_to_close_rcv_fds 內容 close fd.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_close_rcv_fds" << std::endl;

        std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_wait_to_close_rcv_fds.begin();
        while (it != ctx_ptr->ctx_list_wait_to_close_rcv_fds.end())
        {
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif

          // Begin: 將 g_fd_table 重新初始化.
          {
            g_fd_table[*it].fd_init(true);
            //std::cout << __func__ << ":" << __LINE__ << " fd_init()-fd:" << *it << std::endl;
          }
          // End: 將 g_fd_table 重新初始化.

          // closd fd.
          mws_close(*it);
          if (g_mws_log_level >= 1)
          {
            std::string log_body = "close rcv fd: " + std::to_string(*it);
            write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_unlock();
          #endif

          // 處理下一筆資料.
          ++it;
        }
        // 清除 ctx_list_wait_to_close_rcv_fds 的全部內容.
        ctx_ptr->ctx_list_wait_to_close_rcv_fds.clear();
      }
      // End: 6. 依照 ctx_list_wait_to_close_rcv_fds 內容 close fd.

      // Begin: 7. 依照 ctx_list_wait_to_check_topic_src_conn_session 內容中的 status 執行發送 0xFE 到 rcv.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " src topic check" << std::endl;

        std::deque<wait_to_check_topic_src_conn_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.begin();
        while (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
        {
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
          // 因為 lock 到 unlock 的過程中, iterator 指向的 fd 可能會改變, 所以必須要記下 fd.
          fd_t lock_fd = it->fd;
          #if (MWS_DEBUG == 1)
            g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[lock_fd].fd_lock();
          #endif

          std::string topic_name = g_fd_table[lock_fd].src_conn_ptr->topic_name;

          if (g_fd_table[it->fd].status == FD_STATUS_SRC_CONN_PREPARE)
          {
            update_g_fd_table_status(it->fd,
                                     FD_STATUS_SRC_CONN_WAIT_FFFD,
                                     __func__,
                                     __LINE__);
            // 換下一筆資料.
            if (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
            {
              ++it;
            }
          }
          else if ((g_fd_table[it->fd].status == FD_STATUS_UNKNOWN) ||
                   (g_fd_table[it->fd].status == FD_STATUS_SRC_CONN_FD_FAIL) ||
                   (g_fd_table[it->fd].status == FD_STATUS_SRC_CONN_WAIT_TO_CLOSE))
          {
            // 換下一筆資料.
            if (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
            {
              ++it;
            }
          }
          else if (g_fd_table[it->fd].src_conn_sent_FC == false)
          {
            //std::cout << __func__ << ":" << __LINE__ << " send fe fd:" << it->fd << std::endl;

            // 以 it->fd 送 0xFE 給 rcv.
            char send_buff[1];
            send_buff[0] = (char)0xFE;
            ssize_t rtv = send_topic_check_code((void*)&send_buff[0], it->fd, 1);
            if (rtv < 0)
            {
              //std::cout << __func__ << ":" << __LINE__ << " send fe fail  fd:" << it->fd << std::endl;

              std::string log_body;
              log_body = "send_topic_check_code(0xFE) error fd: " + std::to_string(it->fd);
              write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

              // fd 發生問題, 要斷線.
              step_send_fe_error(it, __func__, __LINE__);
            }
            else
            {
              //std::cout << __func__ << ":" << __LINE__ << " send fe ok fd:" << it->fd << std::endl;
              //sleep(1);

              // 換下一筆資料.
              if (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
              {
                ++it;
              }
            }
          } // if (g_fd_table[it->fd].src_conn_sent_FC == false)
          else
          {
            // 換下一筆資料.
            if (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
            {
              ++it;
            }
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[lock_fd].fd_unlock();
          #endif
        } // while (it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end())
      }
      // End: 7. 依照 ctx_list_wait_to_check_topic_src_conn_session 內容中的 status 執行發送 0xFE 到 rcv.

      // Begin: 8. 依照 ctx_list_wait_to_check_topic_rcv_session 內容中的 status 執行發送 0xFF 到 src conn.
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " rcv topic check ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.size = " << ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.size() << std::endl;

        std::deque<wait_to_check_topic_rcv_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.begin();
        while (it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end())
        {
          //std::cout << __func__ << ":" << __LINE__ << " rcv topic check" << std::endl;
          //sleep(5);
          // 因為 lock 到 unlock 的過程中, iterator 指向的 fd 可能會改變, 所以必須要記下 fd.
          fd_t lock_fd = it->fd;
          #if (MWS_DEBUG == 1)
            g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[lock_fd].fd_lock();
          #endif

          std::string topic_name = g_fd_table[lock_fd].rcv_ptr->topic_name;

          if (g_fd_table[it->fd].status == FD_STATUS_RCV_PREPARE)
          {
            // 開始起算 topic check timeout 時間.
            it->t_starting_time = t_current;

            //std::cout << __func__ << ":" << __LINE__ << " FD_STATUS_RCV_PREPARE" << std::endl;
            update_g_fd_table_status(it->fd,
                                     FD_STATUS_RCV_WAIT_FEFC,
                                     __func__,
                                     __LINE__);
            // 換下一筆資料.
            if (it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end())
            {
              ++it;
            }
          }
          else if ((g_fd_table[it->fd].status == FD_STATUS_UNKNOWN) ||
                   (g_fd_table[it->fd].status == FD_STATUS_RCV_FD_FAIL) ||
                   (g_fd_table[it->fd].status == FD_STATUS_RCV_WAIT_TO_CLOSE))
          {
            //std::cout << __func__ << ":" << __LINE__ << " next fd" << std::endl;
            // 換下一筆資料.
            if (it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end())
            {
              ++it;
            }
          }
          else if (g_fd_table[it->fd].rcv_sent_FD == false)
          {
            //std::cout << __func__ << ":" << __LINE__ << " send FF fd: " << it->fd << std::endl;

            // 檢查 topic check 作業是否 timeout.
            if ((t_current - it->t_starting_time) > RCV_TOPIC_CHECK_TIMED_OUT_SEC)
            {
              {
                std::string log_body = "t_current: " + std::to_string(t_current) +
                                       ", topic check starting time:" + std::to_string(it->t_starting_time) +
                                       ", RCV_TOPIC_CHECK_TIMED_OUT_SEC:" + std::to_string(RCV_TOPIC_CHECK_TIMED_OUT_SEC);
                write_to_log(g_fd_table[it->fd].rcv_ptr->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }

              std::string log_body = "topic check of rcv fd: " + std::to_string(it->fd) + " is timed out.";
              write_to_log(g_fd_table[it->fd].rcv_ptr->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

              rcv_topic_check_timeout_error(it->fd, __func__, __LINE__);
            }
            else
            {
              // 以 it->fd 送 0xFF 給 src conn.
              char send_buff[1];
              send_buff[0] = (char)0xFF;
              ssize_t rtv = send_topic_check_code((void*)&send_buff[0], it->fd, 1);
              if (rtv < 0)
              {
                //std::cout << __func__ << ":" << __LINE__ << " send_topic_check_code() rtv < 0 " << std::endl;
  
                std::string log_body;
                log_body = "send_topic_check_code(0xFF) error fd: " + std::to_string(it->fd);
                write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
  
                //std::cout << __func__ << ":" << __LINE__ << " send ff error fd:" << it->fd << std::endl;
                //sleep(5);
                // fd 發生問題, 要斷線.
                step_send_ff_error(it, __func__, __LINE__);
              }
              else
              {
                //std::cout << __func__ << ":" << __LINE__ << " send FF ok fd:" << it->fd << std::endl;
                //sleep(5);
                // 換下一筆資料.
                if (it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end())
                {
                  ++it;
                }
              }
            }
          } // if (g_fd_table[it->fd].rcv_sent_FD == false)
          else
          {
            //std::cout << __func__ << ":" << __LINE__ << " next" << std::endl;

            // 檢查 topic check 作業是否 timeout.
            if ((t_current - it->t_starting_time) > RCV_TOPIC_CHECK_TIMED_OUT_SEC)
            {
              {
                std::string log_body = "t_current: " + std::to_string(t_current) +
                                       ", topic check starting time:" + std::to_string(it->t_starting_time) +
                                       ", RCV_TOPIC_CHECK_TIMED_OUT_SEC:" + std::to_string(RCV_TOPIC_CHECK_TIMED_OUT_SEC);
                write_to_log(g_fd_table[it->fd].rcv_ptr->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }

              std::string log_body = "topic check of rcv fd: " + std::to_string(it->fd) + " is timed out.";
              write_to_log(g_fd_table[it->fd].rcv_ptr->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

              rcv_topic_check_timeout_error(it->fd, __func__, __LINE__);
            }
            else
            {
              // 換下一筆資料.
              if (it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end())
              {
                ++it;
              }
            }
          }

          #if (MWS_DEBUG == 1)
            g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[lock_fd].fd_unlock();
          #endif
        }
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " rcv topic check" << std::endl;
      }
      // End: 8. 依照 ctx_list_wait_to_check_topic_rcv_session 內容中的 status 執行發送 0xFF 到 src conn.

      // Begin: 9. 依照 ctx_list_wait_to_connect_rcv_session 內容執行 rcv 連線到 src.
      // 1. 取得 conn_fd.
      // 2. connect to src.
      // 3. 將完成連線的設定從 ctx_list_wait_to_connect_rcv_session 移除.
      {
        //std::cout << __func__ << ":" << __LINE__ << " ctx_list_wait_to_connect_rcv_session.size():" << std::to_string(ctx_ptr->ctx_list_wait_to_connect_rcv_session.size()) << std::endl;

        std::deque<wait_to_connect_rcv_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_connect_rcv_session.begin();
        while (it != ctx_ptr->ctx_list_wait_to_connect_rcv_session.end())
        {
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
          if ((it->try_cnt < 10) || (flag_time_to_reconnect == true))
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            step_rcv_connect(it, ctx_ptr, std::string(__func__), __LINE__);
          }
          else
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            ++it;
          }
        } // while (it != ctx_ptr->ctx_list_wait_to_connect_rcv_session.end())
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " ctx_list_wait_to_connect_rcv_session" << std::endl;
      }
      // End: 9. 依照 ctx_list_wait_to_connect_rcv_session 內容執行 rcv 連線到 src.
    }
    // End: 每間隔 t_prev_batch_job 秒, 批次執行的工作的區段.

    // Begin: using select() to monitor file descriptors.
    {
      // select timeout value.
      // Notic: Upon successful completion, the select() function
      //        may modify the object pointed to by the timeout argument.
      select_timeout.tv_sec = 0;
      select_timeout.tv_usec = 5000;

      ctx_ptr->rset = ctx_ptr->all_set;
      int ready_fd_cnt = mws_select((ctx_ptr->max_fd + 1),
                                    &ctx_ptr->rset,
                                    NULL,
                                    NULL,
                                    &select_timeout);
      // Begin: debug.
      //{
      //  if (ready_fd_cnt > 0)
      //  {
      //    std::cout << __func__ << ":" << __LINE__ << " ready_fd_cnt:" << ready_fd_cnt << std::endl;
      //    sleep(3);
      //  }
      //}
      // End: debug.
      if (ready_fd_cnt > 0)
      {
        //std::cout << __func__ << ":" << __LINE__ << " ready_fd_cnt > 0, ctx_ptr->max_fd = " << ctx_ptr->max_fd << std::endl;
        // Begin: 處理 src listen fds (新連線).
        {
          //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock();
          #endif

          std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_src_listen_fds.begin();
          while (it != ctx_ptr->ctx_list_owned_src_listen_fds.end())
          {
            if (FD_ISSET(*it, &ctx_ptr->rset))
            {
              // 表示有 rcv 連線到這個 src listen fd.
              //std::cout << __func__ << ":" << __LINE__ << " begin select listen fd" << std::endl;
              // step_accept_connection() 已經完成維護各變數和刷 log 之工作.
              if (g_fd_table[*it].status != FD_STATUS_SRC_LISTEN_WAIT_TO_CLOSE)
              {
                step_accept_connection(ctx_ptr, *it);
              }
              //std::cout << __func__ << ":" << __LINE__ << " end select listen fd (accept ok)" << std::endl;
            }
            if (it != ctx_ptr->ctx_list_owned_src_listen_fds.end())
            {
              ++it;
            }
          }

          //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock();
          #endif
        }
        // End: 處理 src listen fds (新連線).
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " end select listen fd" << std::endl;
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " before select src conn fd" << std::endl;
        // Begin: 處理 src conn fds (src 收到資料).
        {
          //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock();
          #endif

          std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_src_conn_fds.begin();
          while (it != ctx_ptr->ctx_list_owned_src_conn_fds.end())
          {
            if (FD_ISSET(*it, &ctx_ptr->rset))
            {
              //std::cout << __func__ << ":" << __LINE__ << " src conn fd" << std::endl;

              if ((g_fd_table[*it].status != FD_STATUS_UNKNOWN) &&
                  (g_fd_table[*it].status != FD_STATUS_SRC_CONN_FD_FAIL) &&
                  (g_fd_table[*it].status != FD_STATUS_SRC_CONN_WAIT_TO_CLOSE))
              {
                // 表示有 rcv 送 message 到這個 src conn fd.
                switch (g_fd_table[*it].status)
                {
                  case FD_STATUS_SRC_CONN_PREPARE:
                  {
                    //std::cout << __func__ << ":" << __LINE__ << " FD_STATUS_SRC_CONN_PREPARE" << std::endl;
                    break;
                  }
                  case FD_STATUS_SRC_CONN_WAIT_FFFD:
                  case FD_STATUS_SRC_CONN_WAIT_FD:
                  {
                    //std::cout << __func__ << ":" << __LINE__ << " bf step_src_conn_wait_fffd" << std::endl;

                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_src_conn_wait_fffd(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << __func__ << ":" << __LINE__ << " af step_src_conn_wait_fffd" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_SRC_CONN_WAIT_TOPIC_NAME:
                  {
                    //std::cout << __func__ << ":" << __LINE__ << " bf step_src_conn_wait_topic_name" << std::endl;

                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_src_conn_wait_topic_name(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << __func__ << ":" << __LINE__ << " af step_src_conn_wait_topic_name" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_SRC_CONN_TOPIC_CHECK_OK:
                  {
                    break;
                  }
                  case FD_STATUS_SRC_CONN_READY:
                  {
                    //std::cout << __func__ << ":" << __LINE__ << " bf step_src_conn_ready" << std::endl;

                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_src_conn_ready(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << __func__ << ":" << __LINE__ << " af step_src_conn_ready" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_SRC_CONN_FD_FAIL:
                  case FD_STATUS_SRC_CONN_WAIT_TO_CLOSE:
                  {
                    break;
                  }

                  default:
                  {
                    break;
                  }
                } // switch (g_fd_table[*it].status)
              } // if ((g_fd_table[*it].status != FD_STATUS_UNKNOWN) &&
                //     (g_fd_table[*it].status != FD_STATUS_SRC_CONN_FD_FAIL) &&
                //     (g_fd_table[*it].status != FD_STATUS_SRC_CONN_WAIT_TO_CLOSE))
            } // if (FD_ISSET(*it, &ctx_ptr->rset))

            if (it != ctx_ptr->ctx_list_owned_src_conn_fds.end())
            {
              ++it;
            }
          }

          //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock();
          #endif
        }
        // End: 處理 src conn fds (src 收到資料).
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " end select src conn fd" << std::endl;
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " before select rcv fd" << std::endl;
        // Begin: 處理 rcv fds (rcv 收到資料).
        {
          //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock();
          #endif

          std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_rcv_fds.begin();
          while (it != ctx_ptr->ctx_list_owned_rcv_fds.end())
          {
            if (FD_ISSET(*it, &ctx_ptr->rset))
            {
              //std::cout << __func__ << ":" << __LINE__ << " rcv fd" << std::endl;

              if ((g_fd_table[*it].status != FD_STATUS_UNKNOWN) &&
                  (g_fd_table[*it].status != FD_STATUS_RCV_FD_FAIL) &&
                  (g_fd_table[*it].status != FD_STATUS_RCV_WAIT_TO_CLOSE))
              {
                // 表示有 src 送 message 到這個 rcv fd.
                switch (g_fd_table[*it].status)
                {
                  case FD_STATUS_RCV_PREPARE:
                  {
                    break;
                  }
                  case FD_STATUS_RCV_WAIT_FEFC:
                  case FD_STATUS_RCV_WAIT_FC:
                  {
                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf step_rcv_wait_fefc" << std::endl;
                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_rcv_wait_fefc(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af step_rcv_wait_fefc" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_RCV_WAIT_TOPIC_NAME:
                  {
                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf step_rcv_wait_topic_name" << std::endl;

                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_rcv_wait_topic_name(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af step_rcv_wait_topic_name" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_RCV_TOPIC_CHECK_OK:
                  {
                    break;
                  }
                  case FD_STATUS_RCV_READY:
                  {
                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf step_rcv_ready" << std::endl;

                    fd_t lock_fd = *it;
                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_lock();
                    #endif

                    step_rcv_ready(it);

                    #if (MWS_DEBUG == 1)
                      g_fd_table[lock_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                    #else
                      g_fd_table[lock_fd].fd_unlock();
                    #endif

                    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af step_rcv_ready" << std::endl;
                    //sleep(5);
                    break;
                  }
                  case FD_STATUS_RCV_FD_FAIL:
                  case FD_STATUS_RCV_WAIT_TO_CLOSE:
                  {
                    break;
                  }
                  default:
                  {
                    break;
                  }
                } // switch (g_fd_table[*it].status)
              } // if ((g_fd_table[*it].status != FD_STATUS_UNKNOWN) &&
                //     (g_fd_table[*it].status != FD_STATUS_RCV_FD_FAIL) &&
                //     (g_fd_table[*it].status != FD_STATUS_RCV_WAIT_TO_CLOSE))
            } // if (FD_ISSET(*it, &ctx_ptr->rset))
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " in rcv fd" << std::endl;
            if (it != ctx_ptr->ctx_list_owned_rcv_fds.end())
            {
              ++it;
            }
          } // while (it != ctx_ptr->ctx_list_owned_rcv_fds.end())
          //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
          #if (MWS_DEBUG == 1)
            ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock();
          #endif
        }
        // End: 處理 rcv fds (rcv 收到資料).
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " end select rcv fd" << std::endl;
      }
      else if (ready_fd_cnt == (-1))
      {
        log_body = "mws_select() failed (rtv: " +
                   std::to_string(ready_fd_cnt) +
                   ", errno: " + std::to_string(errno) +
                   ", strerr: " + strerror(errno) + ")";
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
        // DEBUG =========================
        //std::cout << "[SELECT FAIL] errno: " << errno << "(" << strerror(errno) << ")" << std::endl;
        // ===============================

        return NULL;
      } // else if (ready_fd_cnt == (-1))
      //else if (ready_fd_cnt == 0)
      //{
        // The return value may be zero if the timeout expired before any file descriptors became ready.
        //std::cout << __func__ << ":" <<__LINE__ << " select timed out" << std::endl;
      //}
    }
    // End: using select() to monitor file descriptors.
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " end select" << std::endl;

    // Begin: signal a condition to unblock dispatch thread(s).
    {
      //pthread_mutex_lock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_lock();
      #endif

      for (std::deque<mws_evq_id_t>::iterator it = g_alive_evq.begin();
           it != g_alive_evq.end();
           ++it)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " Begin SIGNAL" << std::endl;
        if (it->evq_ptr->flag_must_unlock == true)
        {
          #if (MWS_DEBUG == 1)
            it->evq_ptr->evq_cond_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            it->evq_ptr->evq_cond_lock();
          #endif

          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
          pthread_cond_signal(&(it->evq_ptr->cond_select_done));
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;

          #if (MWS_DEBUG == 1)
            it->evq_ptr->evq_cond_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            it->evq_ptr->evq_cond_unlock();
          #endif
        }
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " End SIGNAL" << std::endl;
        //sleep(3);
      }
      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif
    }
    // End: signal a condition to unblock dispatch thread(s).

    #ifdef __TANDEM
      //sched_yield();
      // 降低 NSK cpu 使用量.
      usleep(10);
    #endif
  }

  ctx_ptr->is_ctx_thread_running = false;

  return NULL;
}
