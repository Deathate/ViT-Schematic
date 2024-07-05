//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_EVQ_CPP 1

#include <ctime>
#include <stdint.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <sched.h>  // sched_yield().
#include <string.h>
#include <sys/socket.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <algorithm>  // find().

#include "../inc/mws_init.h"
#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_socket.h"
#include "../inc/mws_time.h"
#include "../inc/mws_type_definition.h"
#include "../inc/mws_util.h"

using namespace mws_global_variable;
using namespace mws_log;
using namespace std;

mws_evq_attr::mws_evq_attr(std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  this->cfg_section = cfg_section;

  // evq set true from default.
  this->is_auto_dispatch = true;

  std::map<std::string, std::string> my_cfg;
  std::string default_section = "default_event_queue_config_value";
  std::map<std::string, std::map<std::string, std::string> >::iterator it;
  it = g_config_mapping.find(default_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定是否為 auto dispatch.
    std::string name("is_auto_dispatch");
    if (my_cfg[name] == "N")
    {
      this->is_auto_dispatch = false;
    }
    if (my_cfg[name] == "Y")
    {
      this->is_auto_dispatch = true;
    }
  }

  it = g_config_mapping.find(cfg_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定是否為 auto dispatch.
    std::string name("is_auto_dispatch");
    if (my_cfg[name] == "N")
    {
      this->is_auto_dispatch = false;
    }
    if (my_cfg[name] == "Y")
    {
      this->is_auto_dispatch = true;
    }
  }

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return;
}

mws_evq_attr::~mws_evq_attr()
{
  return;
}

void mws_evq_attr::mws_modify_evq_attr(std::string attr_name,
                                       std::string attr_value)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  if (attr_name == "is_auto_dispatch")
  {
    if (attr_value == "N")
    {
      this->is_auto_dispatch = false;
    }
    if (attr_value == "Y")
    {
      this->is_auto_dispatch = true;
    }
  }

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return;
}

int32_t mws_init_evq(mws_evq_t* evq_ptr,
                     const bool is_from_cfg,
                     const mws_evq_attr_t mws_evq_attr,
                     const std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  evq_ptr->object_status = 0;

  if (is_from_cfg == false)
  {
    evq_ptr->cfg_section = mws_evq_attr.cfg_section;
    evq_ptr->is_auto_dispatch = mws_evq_attr.is_auto_dispatch;
  } // if (is_from_cfg == false)
  else
  {
    evq_ptr->cfg_section = cfg_section;
    // Begin: evq set from default.
    evq_ptr->is_auto_dispatch = true;
    // End: evq set from default.
    // Begin: 從 cfg 的 default 取得設定值.
    std::map<std::string, std::string> my_cfg;
    std::string default_section = "default_event_queue_config_value";
    std::map<std::string, std::map<std::string, std::string> >::iterator it;
    it = g_config_mapping.find(default_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定是否為 auto dispatch.
      std::string name("is_auto_dispatch");
      if (my_cfg[name] == "N")
      {
        evq_ptr->is_auto_dispatch = false;
      }
      if (my_cfg[name] == "Y")
      {
        evq_ptr->is_auto_dispatch = true;
      }
    }
    // End: 從 cfg 的 default 取得設定值.
    // Begin: 從設定的 cfg section 取得設定值.
    it = g_config_mapping.find(cfg_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定是否為 auto dispatch.
      std::string name("is_auto_dispatch");
      if (my_cfg[name] == "N")
      {
        evq_ptr->is_auto_dispatch = false;
      }
      if (my_cfg[name] == "Y")
      {
        evq_ptr->is_auto_dispatch = true;
      }
    }
    // End: 從設定的 cfg section 取得設定值.
  } // else of if (is_from_cfg == false)

  evq_ptr->evq_no = g_num_of_evq++;

  // 建立屬於此 evq 的 timer_callback 工具物件.
  evq_ptr->timer_callback_ptr = new mws_timer_callback_t(false);

  evq_ptr->mut_data_maintain = PTHREAD_MUTEX_INITIALIZER;
  evq_ptr->mut_select_done = PTHREAD_MUTEX_INITIALIZER;
  evq_ptr->cond_select_done = PTHREAD_COND_INITIALIZER;

  evq_ptr->flag_must_unlock = false;

  #if (MWS_DEBUG == 1)
    evq_ptr->prev_check_time = time(NULL);
  #endif

  mws_evq_id_t temp_obj;
  temp_obj.evq_no = evq_ptr->evq_no;
  temp_obj.evq_ptr = evq_ptr;
  g_alive_evq.push_back(temp_obj);

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return 0;
}

mws_evq::mws_evq(mws_evq_attr_t mws_evq_attr)
{
  // 無用但需要的變數.
  std::string cfg_section("");
  int32_t rtv = mws_init_evq(this,
                             false,
                             mws_evq_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_evq constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_evq constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_evq::mws_evq(std::string cfg_section)
{
  // 無用但需要的變數.
  mws_evq_attr_t mws_evq_attr("");
  int32_t rtv = mws_init_evq(this,
                             true,
                             mws_evq_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_evq constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_evq constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_evq::~mws_evq()
{
  // 等待所有屬於此 evq 的 fd 都結束工作才能開始解構此 evq.
  while (this->evq_list_owned_fds.size() != 0)
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " evq_list_owned_fds.size():" << std::to_string(this->evq_list_owned_fds.size()) << std::endl;
    sleep(1);
  }

  // Begin: 停止 dispatch thread.
  {
    this->is_auto_dispatch = false;

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf pthread_mutex_lock" << std::endl;

    #if (MWS_DEBUG == 1)
      this->evq_cond_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->evq_cond_lock();
    #endif

    pthread_cond_signal(&this->cond_select_done);

    #if (MWS_DEBUG == 1)
      this->evq_cond_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->evq_cond_unlock();
    #endif

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af pthread_mutex_lock" << std::endl;

    while (this->is_dispatch_thread_running == true)
    {
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " while (this->is_dispatch_thread_running == true)" << std::endl;
      sleep(1);
      //usleep(1000);
    }
  }
  // End: 停止 dispatch thread.

  // 消滅 timer_callback 工具物件.
  delete this->timer_callback_ptr;
  this->timer_callback_ptr = NULL;

  // Begin: destroy mut_data_maintain.
  {
    int rtv = pthread_mutex_destroy(&(this->mut_data_maintain));
    if (rtv != 0)
    {
      std::string log_body = "pthread_mutex_destroy() failed (rtv: " + std::to_string(rtv) +
                             ", errno: " + std::to_string(errno) +
                             ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: destroy mut_data_maintain.

  // Begin: destroy mut_select_done.
  {
    int rtv = pthread_mutex_destroy(&(this->mut_select_done));
    if (rtv != 0)
    {
      std::string log_body = "pthread_mutex_destroy() failed (rtv: " + std::to_string(rtv) +
                             ", errno: " + std::to_string(errno) +
                             ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: destroy mut_select_done.

  // Begin: 刪除 g_alive_evq 中的資料.
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
      if (it->evq_ptr == this)
      {
        g_alive_evq.erase(it);

        //pthread_mutex_unlock(&g_mws_global_mutex);
        #if (MWS_DEBUG == 1)
          g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_mws_global_mutex_unlock();
        #endif

        break; // for (std::deque<mws_evq_id_t>::iterator it = g_alive_evq.begin();
               //      it != g_alive_evq.end();
               //      ++it)
      }
    }
    //pthread_mutex_unlock(&g_mws_global_mutex);
    #if (MWS_DEBUG == 1)
      g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_mws_global_mutex_unlock();
    #endif
  }
  // End: 刪除 g_alive_evq 中的資料.

  std::string log_body = "mws_evq destructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

mws_event_t* mws_evq::create_non_msg_event(fd_t fd, uint8_t event_type, bool need_to_lock_fd)
{
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " start create_non_msg_event" << std::endl;

  mws_event_t* event_ptr = new mws_event_t;

  switch (event_type)
  {
    // FD_ROLE_SRC_CONN
    case MWS_SRC_EVENT_CONNECT:
    case MWS_SRC_EVENT_DISCONNECT:
    {
      if (need_to_lock_fd == true)
      {
        #if (MWS_DEBUG == 1)
          g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[fd].fd_lock();
        #endif
      }
      event_ptr->fd = fd;
      event_ptr->event_type = event_type;
      event_ptr->topic_name = g_fd_table[fd].src_conn_ptr->topic_name;
      event_ptr->src_ptr = g_fd_table[fd].src_conn_ptr;
      event_ptr->rcv_ptr = NULL;
      // Begin: 更新 event_ptr->src_listen_addr.
      sockaddr_in_t_to_string(g_fd_table[fd].src_conn_listen_addr_info,
                              event_ptr->src_addr.str_ip,
                              event_ptr->src_addr.str_port);
      // End: 更新 event_ptr->src_listen_addr.
      // Begin: 更新 event_ptr->rcv_addr.
      sockaddr_in_t_to_string(g_fd_table[fd].src_conn_rcv_addr_info,
                              event_ptr->rcv_addr.str_ip,
                              event_ptr->rcv_addr.str_port);
      // End: 更新 event_ptr->rcv_addr.
      event_ptr->seq_num = 0;
      event_ptr->loss_num = 0;
      event_ptr->msg_size = 0;
      if (need_to_lock_fd == true)
      {
        #if (MWS_DEBUG == 1)
          g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[fd].fd_unlock();
        #endif
      }
      break;
    }
    // FD_ROLE_RCV
    case MWS_MSG_BOS:
    case MWS_MSG_EOS:
    {
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
      if (need_to_lock_fd == true)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
        #if (MWS_DEBUG == 1)
          g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[fd].fd_lock();
        #endif
      }
      event_ptr->fd = fd;
      event_ptr->event_type = event_type;
      event_ptr->topic_name = g_fd_table[fd].rcv_ptr->topic_name;
      event_ptr->src_ptr = NULL;
      event_ptr->rcv_ptr = g_fd_table[fd].rcv_ptr;
      // Begin: 更新 event_ptr->src_listen_addr.
      sockaddr_in_t_to_string(g_fd_table[fd].rcv_listen_addr_info,
                              event_ptr->src_addr.str_ip,
                              event_ptr->src_addr.str_port);
      // End: 更新 event_ptr->src_listen_addr.
      // Begin: 更新 event_ptr->rcv_addr.
      sockaddr_in_t_to_string(g_fd_table[fd].rcv_addr_info,
                              event_ptr->rcv_addr.str_ip,
                              event_ptr->rcv_addr.str_port);
      // End: 更新 event_ptr->rcv_addr.
      event_ptr->seq_num = 0;
      event_ptr->loss_num = 0;
      event_ptr->msg_size = 0;

      if (need_to_lock_fd == true)
      {
        #if (MWS_DEBUG == 1)
          g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[fd].fd_unlock();
        #endif
      }

      break;
    }
    default:
    {
      return NULL;
    }
  }

  this->flag_must_unlock = true;

  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
  return event_ptr;
}

ssize_t mws_evq::push_back_non_msg_event(mws_event_t* event_ptr)
{
  // Begin: lock evq.
  {
    // try lock evq.
    #if (MWS_DEBUG == 1)
      int rtv = this->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      int rtv = this->evq_trylock();
    #endif

    while (rtv == EBUSY)
    {
      // 解除 fd 的 lock.
      #if (MWS_DEBUG == 1)
        g_fd_table[event_ptr->fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[event_ptr->fd].fd_unlock();
      #endif

      usleep(10);

      // try lock evq.
      #if (MWS_DEBUG == 1)
        rtv = this->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        rtv = this->evq_trylock();
      #endif

      if (rtv == 0)
      {
        // lock fd.
        #if (MWS_DEBUG == 1)
          g_fd_table[event_ptr->fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[event_ptr->fd].fd_lock();
        #endif
      }
    }
  }
  // End: lock evq.

  switch (event_ptr->event_type)
  {
    case MWS_SRC_EVENT_CONNECT:
    case MWS_MSG_BOS:
    {
      this->connect_event_queue.push(event_ptr);
      break;
    }
    case MWS_SRC_EVENT_DISCONNECT:
    case MWS_MSG_EOS:
    {
      this->disconnect_event_queue.push(event_ptr);
      break;
    }
    default:
    {
      #if (MWS_DEBUG == 1)
        this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->evq_unlock();
      #endif

      return -1;
    }
  }

  #if (MWS_DEBUG == 1)
    this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->evq_unlock();
  #endif

  return 0;
}

int mws_evq::mws_event_dispatch()
{
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " Start mws_event_dispatch" << std::endl;

  std::string log_body;

  this->is_dispatch_thread_running = true;

  do
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " Wait signal" << std::endl;
    // Begin: WAIT SIGNAL.
    #if (MWS_DEBUG == 1)
      this->evq_cond_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->evq_cond_lock();
    #endif

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;

    #if (MWS_DEBUG == 1)
      this->evq_cond_wait(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->evq_cond_wait();
    #endif

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
    this->flag_must_unlock = false;

    #if (MWS_DEBUG == 1)
      this->evq_cond_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->evq_cond_unlock();
    #endif

    // End: WAIT SIGNAL.
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " Get signal" << std::endl;
    //sleep(2);

    // Begin: 執行 timer callback.
    this->timer_callback_ptr->timer_manager();
    // End: 執行 timer callback.

    // Begin: dispatch all the events that is currently in event queue.
    int rtv = this->dispatch_events();
    if (rtv != 0)
    {
      this->is_dispatch_thread_running = false;
      return rtv;
    }
    // End: dispatch all the events that is currently in event queue.
  }
  while (this->is_auto_dispatch == true);

  this->is_dispatch_thread_running = false;

  return 0;
}

std::string mws_evq::mws_get_cfg_section()
{
  return this->cfg_section;
}

uint32_t mws_evq::mws_get_object_status()
{
  return this->object_status;
}

int mws_evq::erase_evq_list_owned_fds(const fd_t fd)
{
  std::deque<fd_t>::iterator it = this->evq_list_owned_fds.begin();
  while (it != this->evq_list_owned_fds.end())
  {
    if (*it == fd)
    {
      this->evq_list_owned_fds.erase(it);
      return 0;
    }
    else
    {
      if (it != this->evq_list_owned_fds.end())
      {
        ++it;
      }
    }
  }

  return 1;
}

int mws_evq::dispatch_connect_events()
{
  while (this->connect_event_queue.empty() == false)
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " this->connect_event_queue.empty() == false" << std::endl;
    mws_event_t* event_ptr = this->connect_event_queue.front();

    fd_t fd = event_ptr->fd;

    #if (MWS_DEBUG == 1)
      g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[fd].fd_lock();
    #endif

    // Begin: 維護 g_fd_table 和呼叫 callback function, dispatch event.
    {
      if (g_fd_table[fd].role == FD_ROLE_SRC_CONN)
      {
        // Begin: 更新 last_heartbeat_time.
        {
          pthread_mutex_lock(&(g_time_current_mutex));
          if (g_time_current > g_fd_table[fd].last_heartbeat_time)
          {
            g_fd_table[fd].last_heartbeat_time = g_time_current;
          }
          pthread_mutex_unlock(&(g_time_current_mutex));
        }
        // End: 更新 last_heartbeat_time.

        //g_fd_table[fd].status = FD_STATUS_SRC_CONN_READY;
        update_g_fd_table_status(fd,
                                 FD_STATUS_SRC_CONN_READY,
                                 __func__,
                                 __LINE__);
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " fd:" << std::to_string(fd) << " FD_STATUS_SRC_CONN_READY" << std::endl;
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

          // 刪除 event 佔用的記憶體空間.
          delete event_ptr;
          // 把處理過的 event 從 connect_event_queue 中 pop 掉.
          this->connect_event_queue.pop();

          #if (MWS_DEBUG == 1)
            g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[fd].fd_unlock();
          #endif

          return rtv;
        }
      }
      else if (g_fd_table[fd].role == FD_ROLE_RCV)
      {
        // Begin: 更新 last_heartbeat_time.
        {
          pthread_mutex_lock(&(g_time_current_mutex));
          if (g_time_current > g_fd_table[fd].last_heartbeat_time)
          {
            g_fd_table[fd].last_heartbeat_time = g_time_current;
          }
          pthread_mutex_unlock(&(g_time_current_mutex));
        }
        // End: 更新 last_heartbeat_time.

        //g_fd_table[fd].status = FD_STATUS_RCV_READY;
        update_g_fd_table_status(fd,
                                 FD_STATUS_RCV_READY,
                                 __func__,
                                 __LINE__);
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " fd:" << std::to_string(fd) << " FD_STATUS_RCV_READY" << std::endl;
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

          // 刪除 event 佔用的記憶體空間.
          delete event_ptr;
          // 把處理過的 event 從 connect_event_queue 中 pop 掉.
          this->connect_event_queue.pop();

          #if (MWS_DEBUG == 1)
            g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[fd].fd_unlock();
          #endif

          return rtv;
        }
      }
      else
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " error" << std::endl;
        // 刷 error log.
      }
      // 刪除 event 佔用的記憶體空間.
      delete event_ptr;
      // 把處理過的 event 從 connect_event_queue 中 pop 掉.
      this->connect_event_queue.pop();
    }
    // Begin: 維護 g_fd_table 和呼叫 callback function, dispatch event.

    #if (MWS_DEBUG == 1)
      g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[fd].fd_unlock();
    #endif

  } // while (this->connect_event_queue.empty() == false)

  return 0;
}

int mws_evq::dispatch_disconnect_events()
{
  while (this->disconnect_event_queue.empty() == false)
  {
    mws_event_t* event_ptr = this->disconnect_event_queue.front();

    fd_t fd = event_ptr->fd;

    #if (MWS_DEBUG == 1)
      g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[fd].fd_lock();
    #endif

    // Begin: 呼叫 callback function, dispatch event.
    {
      if (g_fd_table[fd].role == FD_ROLE_SRC_CONN)
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

          // 刪除 event 佔用的記憶體空間.
          delete event_ptr;
          // 把處理過的 event 從 disconnect_event_queue 中 pop 掉.
          this->disconnect_event_queue.pop();

          #if (MWS_DEBUG == 1)
            g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[fd].fd_unlock();
          #endif

          return rtv;
        }

        // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
        {
          // 外部已經 lock, 不需要在此 lock.
          //#if (MWS_DEBUG == 1)
          //  g_fd_table[fd].src_conn_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          //#else
          //  g_fd_table[fd].src_conn_ptr->evq_ptr->evq_lock();
          //#endif

          int rtv = g_fd_table[fd].src_conn_ptr->evq_ptr->erase_evq_list_owned_fds(fd);
          if (rtv != 0)
          {
            //std::cout << std::string(__func__) << std::endl;
            // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
            std::string log_body;
            log_body = "fd: " + std::to_string(fd) + " does not exist in evq_list_owned_fds";
            write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
          }

          // 外部已經 lock, 不需要在此 lock, 所以也不用 unlock.
          //#if (MWS_DEBUG == 1)
          //  g_fd_table[fd].src_conn_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          //#else
          //  g_fd_table[fd].src_conn_ptr->evq_ptr->evq_unlock();
          //#endif
        }
        // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

        // 修改 g_fd_table 的 status 為 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE.
        update_g_fd_table_status(fd,
                                 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE,
                                 __func__,
                                 __LINE__);

        // 把 fd 放入 ctx 的 ctx_list_wait_to_close_src_conn_fds.
        g_fd_table[fd].src_conn_ptr->ctx_ptr->ctx_list_wait_to_close_src_conn_fds.push_back(fd);
      }
      else if (g_fd_table[fd].role == FD_ROLE_RCV)
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

          // 刪除 event 佔用的記憶體空間.
          delete event_ptr;
          // 把處理過的 event 從 disconnect_event_queue 中 pop 掉.
          this->disconnect_event_queue.pop();

          #if (MWS_DEBUG == 1)
            g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[fd].fd_unlock();
          #endif

          return rtv;
        }

        // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
        {
          // 外部已經 lock, 不需要在此 lock.
          //#if (MWS_DEBUG == 1)
          //  g_fd_table[fd].rcv_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          //#else
          //  g_fd_table[fd].rcv_ptr->evq_ptr->evq_lock();
          //#endif

          int rtv = g_fd_table[fd].rcv_ptr->evq_ptr->erase_evq_list_owned_fds(fd);
          if (rtv != 0)
          {
            //std::cout << std::string(__func__) << std::endl;
            // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
            std::string log_body;
            log_body = "fd: " + std::to_string(fd) + " does not exist in evq_list_owned_fds";
            write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
          }

          // 外部已經 lock, 不需要在此 lock, 所以也不用 unlock.
          //#if (MWS_DEBUG == 1)
          //  g_fd_table[fd].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          //#else
          //  g_fd_table[fd].rcv_ptr->evq_ptr->evq_unlock();
          //#endif
        }
        // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

        // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_WAIT_TO_CLOSE.
        update_g_fd_table_status(fd,
                                 FD_STATUS_RCV_WAIT_TO_CLOSE,
                                 __func__,
                                 __LINE__);

        // 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
        g_fd_table[fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(fd);
        if (g_mws_log_level >= 1)
        {
          std::string log_body = "push to ctx_list_wait_to_close_rcv_fds-rcv fd: " + std::to_string(fd);
          write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
        }
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " call ctx_list_wait_to_close_rcv_fds.push_back() fd:" << std::to_string(fd) << std::endl;
      }
      else
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " error" << std::endl;
        // 刷 error log.
      }
      // 刪除 event 佔用的記憶體空間.
      delete event_ptr;
      // 把處理過的 event 從 disconnect_event_queue 中 pop 掉.
      this->disconnect_event_queue.pop();
    }
    // Begin: 呼叫 callback function, dispatch event.

    // Begin: maintain g_fd_table.
    {
      if (g_fd_table[fd].role == FD_ROLE_SRC_CONN)
      {
        //g_fd_table[fd].status = FD_STATUS_SRC_CONN_WAIT_TO_CLOSE;
        update_g_fd_table_status(fd,
                                 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE,
                                 __func__,
                                 __LINE__);
      }
      else if (g_fd_table[fd].role == FD_ROLE_RCV)
      {
        //g_fd_table[fd].status = FD_STATUS_RCV_WAIT_TO_CLOSE;
        update_g_fd_table_status(fd,
                                 FD_STATUS_RCV_WAIT_TO_CLOSE,
                                 __func__,
                                 __LINE__);
      }
      else if (g_fd_table[fd].role == FD_ROLE_SRC_LISTEN)
      {
        //g_fd_table[fd].status = FD_STATUS_SRC_LISTEN_WAIT_TO_CLOSE;
        update_g_fd_table_status(fd,
                                 FD_STATUS_SRC_LISTEN_WAIT_TO_CLOSE,
                                 __func__,
                                 __LINE__);
      }
      else
      {
        // 刷 error log.
      }
    }
    // End: maintain g_fd_table.

    #if (MWS_DEBUG == 1)
      g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[fd].fd_unlock();
    #endif
  } // while (this->disconnect_event_queue.empty() == false)

  return 0;
}

// 功能: dispatch event(s).
// 回傳值 0: 表示成功.
// 回傳值 非0: 表示失敗.
// 參數 evq_ptr: 指向所對應的 mws_evq 物件.
int mws_evq::dispatch_events()
{
  #if (MWS_DEBUG == 1)
    this->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->evq_lock();
  #endif

  // Begin: dispatch MWS_SRC_EVENT_CONNECT/MWS_MSG_BOS events.
  {
    int rtv = this->dispatch_connect_events();
    if (rtv != 0)
    {
      #if (MWS_DEBUG == 1)
        this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->evq_unlock();
      #endif

      return rtv;
    }
  }
  // End: dispatch MWS_SRC_EVENT_CONNECT/MWS_MSG_BOS events.

  // Begin: dispatch MWS_SRC_DATA/MWS_MSG_DATA events.
  {
    std::deque<fd_t>::iterator it = this->evq_list_owned_fds.begin();
    while (it != this->evq_list_owned_fds.end())
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[*it].fd_lock();
      #endif

      if ((g_fd_table[*it].status == FD_STATUS_SRC_CONN_READY) || (g_fd_table[*it].status == FD_STATUS_RCV_READY))
      {
        // queue 有資料.
        if (g_fd_table[*it].msg_evq_ptr->empty(g_fd_table[*it].msg_evq_number) == false)
        {
          mws_fast_deque_t* msg_evq_ptr = g_fd_table[*it].msg_evq_ptr;
          ssize_t msg_evq_number = g_fd_table[*it].msg_evq_number;

          // flag that tells if all parsable buffers are parsed.
          bool all_buffers_parsed = false;
          // if all parsable buffers are parsed and no unparsable ones left, then pop all buffers (pop_head_to_current).
          bool pop_to_current = false;

          // *** FRONT (start_front() + at()).
          // current_buffer_ptr points to the block that is currently dealing with.
          mws_msg_evq_buffer_t* current_buffer_ptr = NULL;
          msg_evq_ptr->start_front(msg_evq_number);
          current_buffer_ptr = (mws_msg_evq_buffer_t*)msg_evq_ptr->at(msg_evq_number);

          bool flag_is_head_ready = false;
          bool flag_must_read_next_block = false;
          // *** loop until all parsable buffers are parsed.
          //     (must have at least one buffer to parse, cause !empty).
          do
          {
            flag_must_read_next_block = false;

            // *** check if head is ready.
            if (current_buffer_ptr->data_size >= g_size_of_mws_pkg_head)
            {
              flag_is_head_ready = true;
            }
            else
            {
              flag_is_head_ready = false;
              flag_must_read_next_block = true;
            }

            // Begin: 等待收到完整的 package body 然後做 parse.
            // 如果目前 block 資料包含了一個完整的 package 則可以做 dispatch event.
            if (flag_is_head_ready == true)
            {
              mws_pkg_head_t* temp_head_ptr =
                (mws_pkg_head_t*)(&current_buffer_ptr->buffer[current_buffer_ptr->begin_pos]);
              uint16_t msg_size = ntohs(temp_head_ptr->msg_size);
              // MSG_TYPE_MSG or MSG_TYPE_HB.
              uint8_t msg_type = temp_head_ptr->msg_type;

              if (current_buffer_ptr->data_size >= ((uint64_t)msg_size + g_size_of_mws_pkg_head))
              {
                if (msg_type == MSG_TYPE_MSG)
                {
                  if (g_fd_table[*it].role == FD_ROLE_RCV)
                  {
                    // 建立 event 並把資訊儲存於其中.
                    mws_event_t temp_event;

                    temp_event.fd = *it;
                    temp_event.event_type = MWS_MSG_DATA;
                    temp_event.topic_name = g_fd_table[*it].rcv_ptr->topic_name;
                    temp_event.src_ptr = NULL;
                    temp_event.rcv_ptr = g_fd_table[*it].rcv_ptr;
                    // src_addr.
                    sockaddr_in_t_to_string(g_fd_table[*it].rcv_listen_addr_info,
                                            temp_event.src_addr.str_ip,
                                            temp_event.src_addr.str_port);
                    // rcv_addr.
                    sockaddr_in_t_to_string(g_fd_table[*it].rcv_addr_info,
                                            temp_event.rcv_addr.str_ip,
                                            temp_event.rcv_addr.str_port);
                    // sequence number of message.
                    //temp_event.seq_num = g_endianness_obj.network_to_host_uint64_t(temp_head_ptr->seq_num);
                    // number of loss message.
                    temp_event.loss_num = 0;
                    // message size.
                    temp_event.msg_size = (size_t)msg_size;
                    // message.
                    // memcpy(void *restrict dest, const void *restrict src, size_t n).
                    // dest: 要 dispatch 的 event 物件的 msg 存放區.
                    // src : 目前 buffer(current_buffer_ptr) 的資料開頭的位置.
                    memset((void*)&temp_event.msg, 0x0, MAX_MSG_SIZE);
                    memcpy((void*)&temp_event.msg,
                           (void*)(&current_buffer_ptr->buffer[(current_buffer_ptr->begin_pos + g_size_of_mws_pkg_head)]),
                           (size_t)msg_size);

                    // *** MAINTAIN (delete parsed data).
                    current_buffer_ptr->begin_pos += ((uint64_t)msg_size + g_size_of_mws_pkg_head);
                    current_buffer_ptr->data_size -= ((uint64_t)msg_size + g_size_of_mws_pkg_head);

                    // *** check rcv is_hot_failover_recv_mode.
                    if (g_fd_table[*it].rcv_ptr->is_hot_failover_recv_mode == true)
                    {
                      // sequence number of message.
                      temp_event.seq_num = g_endianness_obj.network_to_host_uint64_t(temp_head_ptr->seq_num);

                      // *** check seq_num.
                      if (temp_event.seq_num == (g_fd_table[*it].rcv_ptr->max_seq_num + 1))
                      {
                        // seq_num 沒有跳號.

                        // 更新 max_seq_num.
                        ++g_fd_table[*it].rcv_ptr->max_seq_num;

                        // Begin: 呼叫 callback function 處理 MWS_MSG_DATA event.
                        {
                          int rtv = (*(g_fd_table[*it].rcv_ptr->cb_ptr))(&temp_event,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_ptr,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_size);
                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for rcv(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.rcv_addr.str_ip + ":" +
                                                   temp_event.rcv_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_MSG_DATA event.
                      }
                      else if (temp_event.seq_num <= g_fd_table[*it].rcv_ptr->max_seq_num)
                      {
                        // 此 seq_num 的資料已經處理過了, 不需要再處理.
                      }
                      else
                      {
                        // seq_num 有跳號.

                        // *** create MWS_MSG_UNRECOVERABLE_LOSS event.
                        mws_event_t temp_event_loss;
                        temp_event_loss.fd = temp_event.fd;
                        temp_event_loss.event_type = MWS_MSG_UNRECOVERABLE_LOSS;
                        temp_event_loss.topic_name = temp_event.topic_name;
                        temp_event_loss.src_ptr = temp_event.src_ptr;
                        temp_event_loss.rcv_ptr = temp_event.rcv_ptr;
                        temp_event_loss.src_addr = temp_event.src_addr;
                        temp_event_loss.rcv_addr = temp_event.rcv_addr;
                        temp_event_loss.seq_num = (g_fd_table[*it].rcv_ptr->max_seq_num + 1);
                        temp_event_loss.loss_num = (temp_event.seq_num - g_fd_table[*it].rcv_ptr->max_seq_num - 1);
                        temp_event_loss.msg_size = 0;

                        // Begin: 呼叫 callback function 處理 MWS_MSG_UNRECOVERABLE_LOSS event.
                        {
                          int rtv = (*(g_fd_table[*it].rcv_ptr->cb_ptr))(&temp_event_loss,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_ptr,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_size);
                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for rcv(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.rcv_addr.str_ip + ":" +
                                                   temp_event.rcv_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_MSG_UNRECOVERABLE_LOSS event.

                        // max_seq_num 直接跳到此次 message 的 seq_num.
                        g_fd_table[*it].rcv_ptr->max_seq_num = temp_event.seq_num;

                        // Begin: 呼叫 callback function 處理 MWS_MSG_DATA event.
                        {
                          int rtv = (*(g_fd_table[*it].rcv_ptr->cb_ptr))(&temp_event,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_ptr,
                                                                         g_fd_table[*it].rcv_ptr->custom_data_size);
                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for rcv(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.rcv_addr.str_ip + ":" +
                                                   temp_event.rcv_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_MSG_DATA event.
                      }
                    } // if (g_fd_table[*it].rcv_ptr->is_hot_failover_recv_mode == true)
                    else
                    {
                      // is_hot_failover_recv_mode is false.

                      // Begin: 呼叫 callback function 處理 MWS_MSG_DATA event.
                      {
                        int rtv = (*(g_fd_table[*it].rcv_ptr->cb_ptr))(&temp_event,
                                                                       g_fd_table[*it].rcv_ptr->custom_data_ptr,
                                                                       g_fd_table[*it].rcv_ptr->custom_data_size);
                        if (rtv != 0)
                        {
                          std::string log_body = "call callback function for rcv(" +
                                                 temp_event.topic_name +
                                                 ", " + temp_event.rcv_addr.str_ip + ":" +
                                                 temp_event.rcv_addr.str_port +
                                                 ") failed (rtv: " + std::to_string(rtv) + ")";
                          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                          #if (MWS_DEBUG == 1)
                            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                          #else
                            g_fd_table[*it].fd_unlock();
                          #endif

                          #if (MWS_DEBUG == 1)
                            this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                          #else
                            this->evq_unlock();
                          #endif

                          return rtv;
                        }
                      }
                      // End: 呼叫 callback function 處理 MWS_MSG_DATA event.
                    } // is_hot_failover_recv_mode is false.
                  } // if (g_fd_table[*it].role == FD_ROLE_RCV)
                  else // else if (g_fd_table[*it].role == FD_ROLE_SRC_CONN)
                  {
                    // FD_ROLE_SRC_CONN.
                    // 建立 event 並把資訊儲存於其中.
                    mws_event_t temp_event;

                    temp_event.fd = *it;
                    temp_event.event_type = MWS_SRC_DATA;
                    temp_event.topic_name = g_fd_table[*it].src_conn_ptr->topic_name;
                    temp_event.src_ptr = g_fd_table[*it].src_conn_ptr;
                    temp_event.rcv_ptr = NULL;
                    // src_addr.
                    sockaddr_in_t_to_string(g_fd_table[*it].src_conn_listen_addr_info,
                                            temp_event.src_addr.str_ip,
                                            temp_event.src_addr.str_port);
                    // rcv_addr.
                    sockaddr_in_t_to_string(g_fd_table[*it].src_conn_rcv_addr_info,
                                            temp_event.rcv_addr.str_ip,
                                            temp_event.rcv_addr.str_port);
                    // sequence number of message.
                    //temp_event.seq_num = g_endianness_obj.network_to_host_uint64_t(temp_head_ptr->seq_num);
                    // number of loss message.
                    temp_event.loss_num = 0;
                    // message size.
                    temp_event.msg_size = (size_t)msg_size;
                    // message.
                    // memcpy(void *restrict dest, const void *restrict src, size_t n).
                    // dest: 要 dispatch 的 event 物件的 msg 存放區.
                    // src : 目前 buffer(current_buffer_ptr) 的資料開頭的位置.
                    memset((void*)&temp_event.msg, 0x0, MAX_MSG_SIZE);
                    memcpy((void*)&temp_event.msg,
                           (void*)(&current_buffer_ptr->buffer[(current_buffer_ptr->begin_pos + g_size_of_mws_pkg_head)]),
                           (size_t)msg_size);

                    // *** MAINTAIN (delete parsed data).
                    current_buffer_ptr->begin_pos += ((uint64_t)msg_size + g_size_of_mws_pkg_head);
                    current_buffer_ptr->data_size -= ((uint64_t)msg_size + g_size_of_mws_pkg_head);

                    if (g_fd_table[*it].src_conn_ptr->is_hot_failover_recv_mode == true)
                    {
                      // sequence number of message.
                      temp_event.seq_num = g_endianness_obj.network_to_host_uint64_t(temp_head_ptr->seq_num);

                      // *** check seq_num.
                      if (temp_event.seq_num == (g_fd_table[*it].src_conn_ptr->max_seq_num + 1))
                      {
                        // seq_num 沒有跳號.

                        // 更新 max_seq_num.
                        ++g_fd_table[*it].src_conn_ptr->max_seq_num;

                        // Begin: 呼叫 callback function 處理 MWS_SRC_DATA event.
                        {
                          int rtv = (*(g_fd_table[*it].src_conn_ptr->cb_ptr))(&temp_event,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_ptr,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_size);

                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for src(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.src_addr.str_ip + ":" +
                                                   temp_event.src_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_SRC_DATA event.
                      }
                      else if (temp_event.seq_num <= g_fd_table[*it].src_conn_ptr->max_seq_num)
                      {
                        // 此 seq_num 的資料已經處理過了, 不需要再處理.
                      }
                      else
                      {
                        // seq_num 有跳號.

                        // *** create MWS_SRC_UNRECOVERABLE_LOSS event.
                        mws_event_t temp_event_loss;
                        temp_event_loss.fd = temp_event.fd;
                        temp_event_loss.event_type = MWS_SRC_UNRECOVERABLE_LOSS;
                        temp_event_loss.topic_name = temp_event.topic_name;
                        temp_event_loss.src_ptr = temp_event.src_ptr;
                        temp_event_loss.rcv_ptr = temp_event.rcv_ptr;
                        temp_event_loss.src_addr = temp_event.src_addr;
                        temp_event_loss.rcv_addr = temp_event.rcv_addr;
                        temp_event_loss.seq_num = (g_fd_table[*it].src_conn_ptr->max_seq_num + 1);
                        temp_event_loss.loss_num = (temp_event.seq_num - g_fd_table[*it].src_conn_ptr->max_seq_num - 1);
                        temp_event_loss.msg_size = 0;

                        // Begin: 呼叫 callback function 處理 MWS_SRC_UNRECOVERABLE_LOSS event.
                        {
                          int rtv = (*(g_fd_table[*it].src_conn_ptr->cb_ptr))(&temp_event_loss,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_ptr,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_size);

                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for src(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.src_addr.str_ip + ":" +
                                                   temp_event.src_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_SRC_UNRECOVERABLE_LOSS event.

                        // max_seq_num 直接跳到此次 message 的 seq_num.
                        g_fd_table[*it].src_conn_ptr->max_seq_num = temp_event.seq_num;

                        // Begin: 呼叫 callback function 處理 MWS_SRC_DATA event.
                        {
                          int rtv = (*(g_fd_table[*it].src_conn_ptr->cb_ptr))(&temp_event,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_ptr,
                                                                              g_fd_table[*it].src_conn_ptr->custom_data_size);

                          if (rtv != 0)
                          {
                            std::string log_body = "call callback function for src(" +
                                                   temp_event.topic_name +
                                                   ", " + temp_event.src_addr.str_ip + ":" +
                                                   temp_event.src_addr.str_port +
                                                   ") failed (rtv: " + std::to_string(rtv) + ")";
                            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                            #if (MWS_DEBUG == 1)
                              g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              g_fd_table[*it].fd_unlock();
                            #endif

                            #if (MWS_DEBUG == 1)
                              this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                            #else
                              this->evq_unlock();
                            #endif

                            return rtv;
                          }
                        }
                        // End: 呼叫 callback function 處理 MWS_SRC_DATA event.
                      }
                    } // if (g_fd_table[*it].src_conn_ptr->is_hot_failover_recv_mode == true)
                    else
                    {
                      // is_hot_failover_recv_mode is false.

                      // Begin: 呼叫 callback function 處理 MWS_SRC_DATA event.
                      {
                        int rtv = (*(g_fd_table[*it].src_conn_ptr->cb_ptr))(&temp_event,
                                                                            g_fd_table[*it].src_conn_ptr->custom_data_ptr,
                                                                            g_fd_table[*it].src_conn_ptr->custom_data_size);

                        if (rtv != 0)
                        {
                          std::string log_body = "call callback function for src(" +
                                                 temp_event.topic_name +
                                                 ", " + temp_event.src_addr.str_ip + ":" +
                                                 temp_event.src_addr.str_port +
                                                 ") failed (rtv: " + std::to_string(rtv) + ")";
                          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                          #if (MWS_DEBUG == 1)
                            g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                          #else
                            g_fd_table[*it].fd_unlock();
                          #endif

                          #if (MWS_DEBUG == 1)
                            this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
                          #else
                            this->evq_unlock();
                          #endif

                          return rtv;
                        }
                      }
                      // End: 呼叫 callback function 處理 MWS_SRC_DATA event.
                    } // is_hot_failover_recv_mode is false.

                  } // FD_ROLE_SRC_CONN.
                } // if (msg_type == MSG_TYPE_MSG)
                else if (msg_type == MSG_TYPE_HB)
                {
                  // Begin: 更新 last_heartbeat_time.
                  {
                    pthread_mutex_lock(&(g_time_current_mutex));
                    if (g_time_current > g_fd_table[*it].last_heartbeat_time)
                    {
                      g_fd_table[*it].last_heartbeat_time = g_time_current;
                    }
                    pthread_mutex_unlock(&(g_time_current_mutex));
                  }
                  // End: 更新 last_heartbeat_time.

                  current_buffer_ptr->begin_pos += g_size_of_mws_pkg_head;
                  current_buffer_ptr->data_size -= g_size_of_mws_pkg_head;
                } // else if (msg_type == MSG_TYPE_HB)
                else // 錯誤的 msg_type.
                {
                  msg_evq_ptr->show_deque_content(msg_evq_number);

                  std::string log_body("");
                  if (g_fd_table[*it].role == FD_ROLE_RCV)
                  {
                    std::string rcv_addr_info_str_ip;
                    std::string rcv_addr_info_str_port;
                    sockaddr_in_t_to_string(g_fd_table[*it].rcv_addr_info,
                                            rcv_addr_info_str_ip,
                                            rcv_addr_info_str_port);
                    log_body = "rcv(" + g_fd_table[*it].rcv_ptr->topic_name + ", " +
                               rcv_addr_info_str_ip + ":" +
                               rcv_addr_info_str_port +
                               ") got msg with unknown event type(" +
                               std::to_string((int)msg_type) + ") with pkg_head content: ";
                  }
                  else // else if (g_fd_table[*it].role == FD_ROLE_SRC_CONN)
                  {
                    std::string src_conn_rcv_addr_info_str_ip;
                    std::string src_conn_rcv_addr_info_str_port;
                    sockaddr_in_t_to_string(g_fd_table[*it].src_conn_rcv_addr_info,
                                            src_conn_rcv_addr_info_str_ip,
                                            src_conn_rcv_addr_info_str_port);
                    log_body = "src(" + g_fd_table[*it].src_conn_ptr->topic_name + ", " +
                               src_conn_rcv_addr_info_str_ip + ":" +
                               src_conn_rcv_addr_info_str_port +
                               ") got msg with unknown event type(" +
                               std::to_string((int)msg_type) + ") with pkg_head content: ";
                  }

                  std::stringstream ss;
                  for (size_t i = 0; i < g_size_of_mws_pkg_head; ++i)
                  {
                    ss << "0x" << std::uppercase << std::setfill('0') << std::setw(2) << std::hex << (int)*(((char*)temp_head_ptr) + i) << " ";
                  }
                  log_body += ss.str();
                  write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

                  exit(1);
                } // 錯誤的 msg_type.
              } // if (current_buffer_ptr->data_size >= ((uint64_t)msg_size + g_size_of_mws_pkg_head))
              else
              {
                flag_must_read_next_block = true;
              }
            } // if (flag_is_head_ready == true)
            // End: 等待收到完整的 package body 然後做 parse.

            if (flag_must_read_next_block == true)
            {
              // 讀取下一個 block.
              if (msg_evq_ptr->read_next(msg_evq_number) == 0)
              {
                // prev_buffer_ptr points to the previous block of the block that is currently dealing with.
                mws_msg_evq_buffer_t* prev_buffer_ptr = current_buffer_ptr;
                current_buffer_ptr = (mws_msg_evq_buffer_t*)(msg_evq_ptr->at(msg_evq_number));

                // *** copy data from prvious buffer to current buffer.
                // memcpy(void *restrict dest, const void *restrict src, size_t n).
                // dest: 下一個 buffer(current_buffer_ptr) 從目前位置往前上一個 buffer(prev_buffer_ptr) 的資料量的位置.
                // src : 上一個 buffer(prev_buffer_ptr) 的資料開頭的位置.
                // len : 上一個 buffer(prev_buffer_ptr) 的資料量大小.
                memcpy((void*)(&current_buffer_ptr->buffer[(current_buffer_ptr->begin_pos - prev_buffer_ptr->data_size)]),
                       (void*)(&prev_buffer_ptr->buffer[prev_buffer_ptr->begin_pos]),
                       prev_buffer_ptr->data_size);

                // *** MAINTAIN (add unparsed data).
                current_buffer_ptr->begin_pos -= prev_buffer_ptr->data_size;
                current_buffer_ptr->data_size += prev_buffer_ptr->data_size;

                //all_buffers_parsed == false;
              } // if (msg_evq_ptr->read_next(msg_evq_number) == 0)
              // current_buffer_ptr 已經是最後一個 block.
              else
              {
                // *** check if data size > 0.
                if (current_buffer_ptr->data_size > 0)
                {
                  // *** pop to prev: 目前的 buffer 還有資料, 需要保留.
                  //pop_to_current = false;
                }
                else // if (current_buffer_ptr->data_size == 0)
                {
                  // *** pop to current: 目前的 buffer 已經沒有資料, 可以 pop 掉.
                  pop_to_current = true;
                }
                all_buffers_parsed = true;
              } // else of if (msg_evq_ptr->read_next(msg_evq_number) == 0)
            } // if (flag_must_read_next_block == true)
          }
          while (all_buffers_parsed == false);

          // *** POP (to prev / to current).
          if (pop_to_current == true)
          {
            msg_evq_ptr->pop_head_to_current(msg_evq_number);
          }
          else
          {
            msg_evq_ptr->pop_head_to_prev(msg_evq_number);
          }
        } // if (g_fd_table[*it].msg_evq_ptr->empty(g_fd_table[*it].msg_evq_number) == false)
        //else
        //{
          // do nothing.
        //}
      } // if ((g_fd_table[*it].status == FD_STATUS_SRC_CONN_READY) || (g_fd_table[*it].status == FD_STATUS_RCV_READY))

      #if (MWS_DEBUG == 1)
        g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[*it].fd_unlock();
      #endif

      if (it != this->evq_list_owned_fds.end())
      {
        ++it;
      }
    } // while (it != this->evq_list_owned_fds.end())
  }
  // End: dispatch MWS_SRC_DATA/MWS_MSG_DATA events.

  // Begin: dispatch MWS_SRC_EVENT_DISCONNECT/MWS_MSG_EOS events.
  {
    int rtv = this->dispatch_disconnect_events();
    if (rtv != 0)
    {
      #if (MWS_DEBUG == 1)
        this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        this->evq_unlock();
      #endif

      return rtv;
    }
  }
  // End: dispatch MWS_SRC_EVENT_DISCONNECT/MWS_MSG_EOS events.

  #if (MWS_DEBUG == 1)
    this->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->evq_unlock();
  #endif

  return 0;
}

#if (MWS_DEBUG == 1)
  void mws_evq::evq_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->mut_data_maintain));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_evq::evq_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->mut_data_maintain));
  }

  void mws_evq::evq_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->mut_data_maintain));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_evq::evq_lock()
  {
    pthread_mutex_lock(&(this->mut_data_maintain));
    return;
  }

  int mws_evq::evq_trylock()
  {
    return pthread_mutex_trylock(&(this->mut_data_maintain));
  }

  void mws_evq::evq_unlock()
  {
    pthread_mutex_unlock(&(this->mut_data_maintain));
    return;
  }
#endif

#if (MWS_DEBUG == 1)
  void mws_evq::evq_cond_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf condition lock evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->mut_select_done));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " condition locking evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  void mws_evq::evq_cond_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->mut_select_done));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " condition unlock evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  void mws_evq::evq_cond_wait(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " enter condition wait evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_cond_wait(&this->cond_select_done, &this->mut_select_done);

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " leave condition wait evq_no:";
      log += std::to_string(this->evq_no);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_evq::evq_cond_lock()
  {
    pthread_mutex_lock(&(this->mut_select_done));

    return;
  }

  void mws_evq::evq_cond_unlock()
  {
    pthread_mutex_unlock(&(this->mut_select_done));

    return;
  }

  void mws_evq::evq_cond_wait()
  {
    pthread_cond_wait(&this->cond_select_done, &this->mut_select_done);

    return;
  }
#endif
