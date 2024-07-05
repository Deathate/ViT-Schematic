//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_REACTOR_ONLY_CTX_CPP 1

#include <iostream>
#include <unistd.h>

#include "../inc/mws_init.h"
#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"

void* reactor_only_ctx_thread_function(void* mws_ctx_ptr);

using namespace mws_global_variable;
using namespace mws_log;

mws_reactor_only_ctx_attr::mws_reactor_only_ctx_attr(std::string cfg_section)
{
  this->cfg_section = cfg_section;

  // ctx set from default.
  this->pthread_stack_size = 0;

  std::map<std::string, std::string> my_cfg;
  std::string default_section = "default_reactor_only_context_config_value";
  std::map<std::string, std::map<std::string, std::string> >::iterator it;
  it = g_config_mapping.find(default_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    //std::cout << "reactor_only_ctx set from default config" << std::endl;
    std::string name("pthread_stack_size");
    this->pthread_stack_size = (long int)atoll(my_cfg[name].c_str());
  }

  it = g_config_mapping.find(cfg_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    //std::cout << "reactor_only_ctx set from config" << std::endl;
    std::string name("pthread_stack_size");
    this->pthread_stack_size = (long int)atoll(my_cfg[name].c_str());
  }

  return;
}

mws_reactor_only_ctx_attr::~mws_reactor_only_ctx_attr()
{
  return;
}

void mws_reactor_only_ctx_attr::mws_modify_reactor_only_ctx_attr(std::string attr_name, std::string attr_value)
{
  if (attr_name == "pthread_stack_size")
  {
    this->pthread_stack_size = (long int)atoll(attr_value.c_str());
    //std::cout << "mws_modify_reactor_only_ctx_attr set: " << this->pthread_stack_size << std::endl;
  }

  return;
}

mws_reactor_only_ctx::mws_reactor_only_ctx(mws_reactor_only_ctx_attr_t mws_reactor_only_ctx_attr)
{
  std::string log_body;

  this->object_status = 0;

  // 建立屬於此 ctx 的 timer_callback 工具物件.
  this->timer_callback_ptr = new mws_timer_callback_t(false);

  this->cfg_section = mws_reactor_only_ctx_attr.cfg_section;

  // ctx set from default.
  ssize_t pthread_stack_size = mws_reactor_only_ctx_attr.pthread_stack_size;

  // 初始化 must_stop_running_ctx_thread.
  this->must_stop_running_ctx_thread = false;
  // 初始化 is_ctx_thread_running.
  this->is_ctx_thread_running = false;

  // Begin: 建立 reactor only ctx thread.
  {
    // 1. 設定 ctx thread 的屬性.
    pthread_attr_t attr;
    int pthread_rtv = 0;

    pthread_rtv = pthread_attr_init(&attr);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_attr_init() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }

    if (pthread_stack_size > 0)
    {
      pthread_rtv = pthread_attr_setstacksize(&attr, pthread_stack_size);
      if (pthread_rtv != 0)
      {
        log_body = "pthread_attr_setstacksize() failed (rtv: " +
                   std::to_string(pthread_rtv) +
                   ", errno: " + std::to_string(errno) +
                   ", strerr: " + strerror(errno) + ")";
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        this->object_status = MWS_ERROR_PTHREAD_CREATE;
        return;
      }
    }

    // 2. 建立 reactor only ctx thread.
    pthread_rtv = pthread_create(&(this->ctx_thread_id),
                                 &attr,
                                 reactor_only_ctx_thread_function,
                                 (void*)this);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_create() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }

    // 3. 消滅 thread 屬性物件.
    pthread_rtv = pthread_attr_destroy(&attr);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_attr_destroy() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }
  }
  // End: 建立 reactor only ctx thread.

  log_body = "mws_reactor_only_ctx constructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

mws_reactor_only_ctx::mws_reactor_only_ctx(std::string cfg_section)
{
  std::string log_body;

  this->object_status = 0;

  // 建立屬於此 ctx 的 timer_callback 工具物件.
  this->timer_callback_ptr = new mws_timer_callback_t(false);

  this->cfg_section = cfg_section;

  // ctx set from default.
  ssize_t pthread_stack_size = 0;

  std::map<std::string, std::string> my_cfg;
  std::string default_section = "default_reactor_only_context_config_value";
  std::map<std::string, std::map<std::string, std::string> >::iterator it;
  it = g_config_mapping.find(default_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    //std::cout << "reactor_only_ctx set from default config" << std::endl;
    std::string name("pthread_stack_size");
    pthread_stack_size = (long int)atoll(my_cfg[name].c_str());
  }

  it = g_config_mapping.find(cfg_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 pthread_stack_size.
    //std::cout << "reactor_only_ctx set from config" << std::endl;
    std::string name("pthread_stack_size");
    pthread_stack_size = (long int)atoll(my_cfg[name].c_str());
  }

  // 初始化 must_stop_running_ctx_thread.
  this->must_stop_running_ctx_thread = false;
  // 初始化 is_ctx_thread_running.
  this->is_ctx_thread_running = false;

  // Begin: 建立 reactor only ctx thread.
  {
    // 1. 設定 ctx thread 的屬性.
    pthread_attr_t attr;
    int pthread_rtv = 0;

    pthread_rtv = pthread_attr_init(&attr);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_attr_init() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }

    if (pthread_stack_size > 0)
    {
      pthread_rtv = pthread_attr_setstacksize(&attr, pthread_stack_size);
      if (pthread_rtv != 0)
      {
        log_body = "pthread_attr_setstacksize() failed (rtv: " +
                   std::to_string(pthread_rtv) +
                   ", errno: " + std::to_string(errno) +
                   ", strerr: " + strerror(errno) + ")";
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        this->object_status = MWS_ERROR_PTHREAD_CREATE;
        return;
      }
    }

    // 2. 建立 reactor only ctx thread.
    pthread_rtv = pthread_create(&(this->ctx_thread_id),
                                 &attr,
                                 reactor_only_ctx_thread_function,
                                 (void*)this);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_create() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }

    // 3. 消滅 thread 屬性物件.
    pthread_rtv = pthread_attr_destroy(&attr);
    if (pthread_rtv != 0)
    {
      log_body = "pthread_attr_destroy() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      this->object_status = MWS_ERROR_PTHREAD_CREATE;
      return;
    }
  }
  // End: 建立 reactor only ctx thread.

  log_body = "mws_reactor_only_ctx constructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

mws_reactor_only_ctx::~mws_reactor_only_ctx()
{
  // Begin: 停止 reactor only ctx thread.
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
  // End: 停止 reactor only ctx thread.

  // 消滅 timer_callback 工具物件.
  delete this->timer_callback_ptr;
  this->timer_callback_ptr = NULL;

  std::string log_body = "mws_reactor_only_ctx destructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

int32_t mws_reactor_only_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
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

int32_t mws_reactor_only_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
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

int32_t mws_reactor_only_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
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

int32_t mws_reactor_only_ctx::mws_schedule_timer(mws_evq_t* evq_ptr,
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

int32_t mws_reactor_only_ctx::mws_cancel_timer(mws_evq_t* evq_ptr,
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

std::string mws_reactor_only_ctx::mws_timer_version(mws_evq_t* evq_ptr)
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

int32_t mws_reactor_only_ctx::mws_show_all_timer_detail(mws_evq_t* evq_ptr)
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

std::string mws_reactor_only_ctx::mws_get_cfg_section()
{
  return this->cfg_section;
}

uint32_t mws_reactor_only_ctx::mws_get_object_status()
{
  return this->object_status;
}

void* reactor_only_ctx_thread_function(void* mws_ctx_ptr)
{
  mws_reactor_only_ctx_t* ctx_ptr = (mws_reactor_only_ctx_t*)mws_ctx_ptr;

  ctx_ptr->is_ctx_thread_running = true;

  while (ctx_ptr->must_stop_running_ctx_thread == false)
  {
    // timer_callback 檢查是否有到期的事件.
    ctx_ptr->timer_callback_ptr->timer_manager();

    usleep(1000);
  }

  ctx_ptr->is_ctx_thread_running = false;

  return NULL;
}
