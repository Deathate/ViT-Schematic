//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_SRC_CPP 1

#include <algorithm>
#include <deque>
#include <fcntl.h>
#include <arpa/inet.h>
#include <iomanip>
#include <iostream>
#include <pthread.h>
#include <sched.h>
#include <sys/socket.h>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <string.h>
#include <netinet/tcp.h>
#include <unistd.h>

#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_init.h"
#include "../inc/mws_log.h"
#include "../inc/mws_socket.h"
#include "../inc/mws_type_definition.h"
#include "../inc/mws_util.h"

using namespace mws_global_variable;
using namespace mws_log;

//extern std::string g_ap_name;

mws_src_attr::mws_src_attr(std::string cfg_section)
{
  this->cfg_section = cfg_section;

  // src set from default.
  this->topic_name = "";
  this->src_ip_port.IP = "127.0.0.1";
  this->src_ip_port.low_port = 1;
  this->src_ip_port.high_port = 65535;
  this->is_hot_failover_recv_mode = false;
  uint16_t temp_listen_port_range_low = 1;
  uint16_t temp_listen_port_range_high = 65535;

  std::map<std::string, std::string> my_cfg;
  std::string temp_topic_name("");
  std::string default_section = "default_source_config_value";
  std::map<std::string, std::map<std::string, std::string> >::iterator it;
  it = g_config_mapping.find(default_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 topic name & check topic size..
    std::string name("topic_name");
    temp_topic_name = std::string(my_cfg[name]);
    // topic size = 0 || topic size > 64 -> error.
    if (!((temp_topic_name.size() == 0) ||
          (temp_topic_name.size() > 64)))
    {
      this->topic_name = temp_topic_name;
    }

    name = "listen_ip";
    if (std::string(my_cfg[name]).size() > 0)
    {
      this->src_ip_port.IP = std::string(my_cfg[name]);
    }
    if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
    {
      temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
    }
    if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
    {
      temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
    }
    set_port_high_low(this->src_ip_port,
                      temp_listen_port_range_low,
                      temp_listen_port_range_high,
                      __FILE__,
                      __func__,
                      __LINE__);

    // 設定是否為 hot failover recv.
    name = "is_hot_failover_recv_mode";
    if (my_cfg[name] == "Y")
    {
      this->is_hot_failover_recv_mode = true;
    }
    if (my_cfg[name] == "N")
    {
      this->is_hot_failover_recv_mode = false;
    }
    //std::cout << "SRC_DEFAUlT is_hot_failover_recv_mode: " << this->is_hot_failover_recv_mode << std::endl;
  }

  it = g_config_mapping.find(cfg_section);
  if ((it != g_config_mapping.end()) && (!it->second.empty()))
  {
    my_cfg = it->second;

    // 設定 topic name & check topic size..
    std::string name("topic_name");
    temp_topic_name = std::string(my_cfg[name]);
    // topic size = 0 || topic size > 64 -> error.
    if (!((temp_topic_name.size() == 0) ||
          (temp_topic_name.size() > 64)))
    {
      this->topic_name = temp_topic_name;
    }

    name = "listen_ip";
    if (std::string(my_cfg[name]).size() > 0)
    {
      this->src_ip_port.IP = std::string(my_cfg[name]);
    }
    if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
    {
      temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
    }
    if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
    {
      temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
    }
    set_port_high_low(this->src_ip_port,
                      temp_listen_port_range_low,
                      temp_listen_port_range_high,
                      __FILE__,
                      __func__,
                      __LINE__);

    // 設定是否為 hot failover recv.
    name = "is_hot_failover_recv_mode";
    if (my_cfg[name] == "Y")
    {
      this->is_hot_failover_recv_mode = true;
    }
    if (my_cfg[name] == "N")
    {
      this->is_hot_failover_recv_mode = false;
    }
    //std::cout << "SRC is_hot_failover_recv_mode: " << this->is_hot_failover_recv_mode << std::endl;
  }

  return;
}

mws_src_attr::~mws_src_attr()
{
  return;
}

void mws_src_attr::mws_modify_src_attr(std::string attr_name, std::string attr_value)
{
  if (attr_name == "topic_name")
  {
    if ((attr_value.size() == 0) ||
        (attr_value.size() > 64))
    {
      return;
    }
    else
    {
      this->topic_name = attr_value;
    }
  }

  if (attr_name == "listen_ip")
  {
    if (std::string(attr_value).size() > 0)
    {
      this->src_ip_port.IP = std::string(attr_value);
    }
  }

  if (attr_name == "listen_port_range_low")
  {
    if (std::string(attr_value).size() > 0)
    {
      if (((uint16_t)atoi(attr_value.c_str()) > 0) && ((uint16_t)atoi(attr_value.c_str()) <= this->src_ip_port.high_port))
      {
        this->src_ip_port.low_port = (uint16_t)atoi(attr_value.c_str());
      }
    }
  }

  if (attr_name == "listen_port_range_high")
  {
    if (std::string(attr_value).size() > 0)
    {
      if (((uint16_t)atoi(attr_value.c_str()) >= this->src_ip_port.low_port) && ((uint16_t)atoi(attr_value.c_str()) < 65536))
      {
        this->src_ip_port.high_port = (uint16_t)atoi(attr_value.c_str());
      }
    }
  }

  if (attr_name == "is_hot_failover_recv_mode")
  {
    if (attr_value == "Y")
    {
      this->is_hot_failover_recv_mode = true;
    }
    if (attr_value == "N")
    {
      this->is_hot_failover_recv_mode = false;
    }
  }

  return;
}

int32_t mws_init_src(mws_src* src_ptr,
                     mws_ctx_t* ctx_ptr,
                     mws_evq_t* evq_ptr,
                     callback_t* src_cb_ptr,
                     void* custom_data_ptr,
                     const size_t custom_data_size,
                     const bool is_from_cfg,
                     const mws_src_attr_t mws_src_attr,
                     const std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  std::string log_body;

  src_ptr->object_status = 0;

  if (is_from_cfg == false)
  {
    src_ptr->cfg_section = mws_src_attr.cfg_section;

    // topic name.
    if ((mws_src_attr.topic_name.size() == 0) ||
        (mws_src_attr.topic_name.size() > 64))
    {
      log_body = "src topic size error or no topic name, topic name: ";
      log_body += mws_src_attr.topic_name;
      log_body += ", size: ";
      log_body += std::to_string(mws_src_attr.topic_name.size());
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      src_ptr->object_status = MWS_ERROR_TOPIC_NAME;

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return -1;
    }
    else
    {
      src_ptr->topic_name = mws_src_attr.topic_name;
    }
    // 接收資料模式.
    // true: hot failover 接收資料模式,
    //       message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
    // false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
    src_ptr->is_hot_failover_recv_mode = mws_src_attr.is_hot_failover_recv_mode;
    // source 的 IP 和 port range.
    src_ptr->src_ip_port = mws_src_attr.src_ip_port;
  } // if (is_from_cfg == false)
  else
  {
    src_ptr->cfg_section = cfg_section;

    // Begin: src set from default.
    src_ptr->topic_name = "";
    src_ptr->src_ip_port.IP = "127.0.0.1";
    src_ptr->src_ip_port.low_port = 1;
    src_ptr->src_ip_port.high_port = 65535;
    src_ptr->is_hot_failover_recv_mode = false;
    uint16_t temp_listen_port_range_low = 1;
    uint16_t temp_listen_port_range_high = 65535;
    // End: src set from default.
    // Begin: 從 cfg 的 default 取得設定值.
    std::map<std::string, std::string> my_cfg;
    std::string temp_topic_name("");
    std::string default_section = "default_source_config_value";
    std::map<std::string, std::map<std::string, std::string> >::iterator it;
    it = g_config_mapping.find(default_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定 topic name & check topic size.
      std::string name("topic_name");
      temp_topic_name = std::string(my_cfg[name]);
      // topic size = 0 || topic size > 64 -> error.
      if (!((temp_topic_name.size() == 0) ||
            (temp_topic_name.size() > 64)))
      {
        src_ptr->topic_name = temp_topic_name;
      }

      name = "listen_ip";
      if (std::string(my_cfg[name]).size() > 0)
      {
        src_ptr->src_ip_port.IP = std::string(my_cfg[name]);
      }
      if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
      {
        temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
      }
      if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
      {
        temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
      }
      set_port_high_low(src_ptr->src_ip_port,
                        temp_listen_port_range_low,
                        temp_listen_port_range_high,
                        __FILE__,
                        __func__,
                        __LINE__);

      // 設定是否為 hot failover recv.
      name = "is_hot_failover_recv_mode";
      if (my_cfg[name] == "Y")
      {
        src_ptr->is_hot_failover_recv_mode = true;
      }
      if (my_cfg[name] == "N")
      {
        src_ptr->is_hot_failover_recv_mode = false;
      }
    }
    // End: 從 cfg 的 default 取得設定值.
    // Begin: 從設定的 cfg section 取得設定值.
    it = g_config_mapping.find(cfg_section);
    if ((it != g_config_mapping.end()) && (!it->second.empty()))
    {
      my_cfg = it->second;

      // 設定 topic name & check topic size.
      std::string name("topic_name");
      temp_topic_name = std::string(my_cfg[name]);
      // topic size = 0 || topic size > 64 -> error.
      if (!((temp_topic_name.size() == 0) ||
            (temp_topic_name.size() > 64)))
      {
        src_ptr->topic_name = temp_topic_name;
      }

      if ((src_ptr->topic_name.size() == 0) ||
          (src_ptr->topic_name.size() > 64))
      {
        log_body = "src topic size error or no topic name, topic name: ";
        log_body += temp_topic_name;
        log_body += ", size: ";
        log_body += std::to_string(temp_topic_name.size());
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        src_ptr->object_status = MWS_ERROR_TOPIC_NAME;

        //pthread_mutex_unlock(&g_mws_global_mutex);
        #if (MWS_DEBUG == 1)
          g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_mws_global_mutex_unlock();
        #endif

        return -1;
      }

      name = "listen_ip";
      if (std::string(my_cfg[name]).size() > 0)
      {
        src_ptr->src_ip_port.IP = std::string(my_cfg[name]);
      }
      if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
      {
        temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
      }
      if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
      {
        temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
      }
      set_port_high_low(src_ptr->src_ip_port,
                        temp_listen_port_range_low,
                        temp_listen_port_range_high,
                        __FILE__,
                        __func__,
                        __LINE__);

      // 設定是否為 hot failover recv.
      name = "is_hot_failover_recv_mode";
      if (my_cfg[name] == "Y")
      {
        src_ptr->is_hot_failover_recv_mode = true;
      }
      if (my_cfg[name] == "N")
      {
        src_ptr->is_hot_failover_recv_mode = false;
      }
    }
    // End: 從設定的 cfg section 取得設定值.
  } // else of if (is_from_cfg == false)

  src_ptr->ctx_ptr = ctx_ptr;
  src_ptr->evq_ptr = evq_ptr;
  src_ptr->max_seq_num = 0;
  src_ptr->cb_ptr = src_cb_ptr;

  // 設定帶入 cb 的資料及資料長度.
  src_ptr->custom_data_size = custom_data_size;
  src_ptr->custom_data_ptr = calloc(1, src_ptr->custom_data_size);
  memcpy(src_ptr->custom_data_ptr, custom_data_ptr, src_ptr->custom_data_size);

  src_ptr->flag_ready_to_release_src = false;

  //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_src_mutex));
  #if (MWS_DEBUG == 1)
    ctx_ptr->ctx_list_owned_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    ctx_ptr->ctx_list_owned_src_mutex_lock();
  #endif

  src_ptr->ctx_ptr->ctx_list_owned_src.push_back(src_ptr);

  //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_src_mutex));
  #if (MWS_DEBUG == 1)
    ctx_ptr->ctx_list_owned_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    ctx_ptr->ctx_list_owned_src_mutex_unlock();
  #endif

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return 0;
}

mws_src::mws_src(mws_src_attr_t mws_src_attr,
                 mws_ctx_t* ctx_ptr,
                 mws_evq_t* evq_ptr,
                 callback_t* src_cb_ptr,
                 void* custom_data_ptr,
                 const size_t custom_data_size)
{
  // 無用但需要的變數.
  std::string cfg_section("");
  int32_t rtv = mws_init_src(this,
                             ctx_ptr,
                             evq_ptr,
                             src_cb_ptr,
                             custom_data_ptr,
                             custom_data_size,
                             false,
                             mws_src_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    int rtv = create_listen_socket(this);
    if (rtv < 0)
    {
      this->object_status = MWS_ERROR_LISTEN_SOCKET_CREATE;
      log_body = "mws_src(" + this->topic_name + ") constructor fail";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
      return;
    }

    // 新增 this->ctx_ptr->ctx_list_owned_src_listen_fds
    //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock();
    #endif

    this->ctx_ptr->ctx_list_owned_src_listen_fds.push_back(this->src_listen_fd);

    //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock();
    #endif

    // 新增 this->src_listen_fd 至 all_set.
    FD_SET(this->src_listen_fd, &this->ctx_ptr->all_set);
    {
      std::string log_body = "Add src listen fd:" +
                             std::to_string(this->src_listen_fd) +
                             " into all_set ";
      write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
    }

    // 更新 ctx::max_fd
    this->ctx_ptr->update_max_fd(this->src_listen_fd);

    // Begin: 更新 g_fd_table.
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[this->src_listen_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[this->src_listen_fd].fd_lock();
      #endif

      g_fd_table[this->src_listen_fd].fd = this->src_listen_fd;
      g_fd_table[this->src_listen_fd].role = FD_ROLE_SRC_LISTEN;
      g_fd_table[this->src_listen_fd].status = FD_STATUS_SRC_LISTEN_READY;
      update_g_fd_table_status(this->src_listen_fd,
                               FD_STATUS_SRC_LISTEN_READY,
                               __func__,
                               __LINE__);
      g_fd_table[this->src_listen_fd].src_listen_ptr = this;
      g_fd_table[this->src_listen_fd].src_listen_addr_info = this->src_listen_addr;

      // Begin: 不使用變數初始化.
      sockaddr_in_t temp_obj;
      memset(&temp_obj, 0, sizeof(temp_obj));
      g_fd_table[this->src_listen_fd].src_conn_ptr = NULL;
      g_fd_table[this->src_listen_fd].src_conn_listen_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].src_conn_rcv_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].src_conn_sent_FC = false;
      g_fd_table[this->src_listen_fd].src_conn_sent_topic_name = false;
      g_fd_table[this->src_listen_fd].rcv_ptr = NULL;
      g_fd_table[this->src_listen_fd].rcv_listen_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].rcv_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].rcv_sent_FD = false;
      g_fd_table[this->src_listen_fd].rcv_sent_topic_name = false;
      // End: 不使用變數初始化.
      #if (MWS_DEBUG == 1)
        g_fd_table[this->src_listen_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[this->src_listen_fd].fd_unlock();
      #endif
    }
    // End: 更新 g_fd_table.

    // src_listen_fd 不用加入 evq_list_owned_fds.

    log_body = "mws_src(" + this->topic_name + ") constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_src(" + this->topic_name + ") constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_src::mws_src(std::string cfg_section,
                 mws_ctx_t* ctx_ptr,
                 mws_evq_t* evq_ptr,
                 callback_t* src_cb_ptr,
                 void* custom_data_ptr,
                 const size_t custom_data_size)
{
  // 無用但需要的變數.
  mws_src_attr_t mws_src_attr("");
  int32_t rtv = mws_init_src(this,
                             ctx_ptr,
                             evq_ptr,
                             src_cb_ptr,
                             custom_data_ptr,
                             custom_data_size,
                             true,
                             mws_src_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    int rtv = create_listen_socket(this);
    if (rtv < 0)
    {
      this->object_status = MWS_ERROR_LISTEN_SOCKET_CREATE;
      log_body = "mws_src(" + this->topic_name + ") constructor fail";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
      return;
    }

    // 新增 this->ctx_ptr->ctx_list_owned_src_listen_fds
    //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_lock();
    #endif

    this->ctx_ptr->ctx_list_owned_src_listen_fds.push_back(this->src_listen_fd);

    //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_owned_src_listen_fds_mutex_unlock();
    #endif

    // 新增 this->src_listen_fd 至 all_set.
    FD_SET(this->src_listen_fd, &this->ctx_ptr->all_set);
    {
      std::string log_body = "Add src listen fd:" +
                             std::to_string(this->src_listen_fd) +
                             " into all_set ";
      write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
    }
    // 更新 ctx::max_fd
    ctx_ptr->update_max_fd(this->src_listen_fd);

    // Begin: 更新 g_fd_table.
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[this->src_listen_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[this->src_listen_fd].fd_lock();
      #endif

      g_fd_table[this->src_listen_fd].fd = this->src_listen_fd;
      g_fd_table[this->src_listen_fd].role = FD_ROLE_SRC_LISTEN;
      //g_fd_table[this->src_listen_fd].status = FD_STATUS_SRC_LISTEN_READY;
      update_g_fd_table_status(this->src_listen_fd,
                               FD_STATUS_SRC_LISTEN_READY,
                               __func__,
                               __LINE__);
      g_fd_table[this->src_listen_fd].src_listen_ptr = this;
      g_fd_table[this->src_listen_fd].src_listen_addr_info = this->src_listen_addr;
      // Begin: 不使用變數初始化.
      sockaddr_in_t temp_obj;
      memset(&temp_obj, 0, sizeof(temp_obj));
      g_fd_table[this->src_listen_fd].src_conn_ptr = NULL;
      g_fd_table[this->src_listen_fd].src_conn_listen_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].src_conn_rcv_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].src_conn_sent_FC = false;
      g_fd_table[this->src_listen_fd].src_conn_sent_topic_name = false;
      g_fd_table[this->src_listen_fd].rcv_ptr = NULL;
      g_fd_table[this->src_listen_fd].rcv_listen_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].rcv_addr_info = temp_obj;
      g_fd_table[this->src_listen_fd].rcv_sent_FD = false;
      g_fd_table[this->src_listen_fd].rcv_sent_topic_name = false;
      // End: 不使用變數初始化.
      #if (MWS_DEBUG == 1)
        g_fd_table[this->src_listen_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[this->src_listen_fd].fd_unlock();
      #endif
    }
    // End: 更新 g_fd_table.

    // src_listen_fd 不用加入 evq_list_owned_fds.

    log_body = "mws_src(" + this->topic_name + ") constructor complete";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_src(" + this->topic_name + ") constructor fail";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_src::~mws_src()
{
  //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_wait_to_stop_src_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock();
  #endif

  this->ctx_ptr->ctx_list_wait_to_stop_src.push_back(this);
  while (this->flag_ready_to_release_src == false)
  {
    //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_wait_to_stop_src_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock();
    #endif

    sleep(1);

    //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_wait_to_stop_src_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_lock();
    #endif
  }
  //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_wait_to_stop_src_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_wait_to_stop_src_mutex_unlock();
  #endif

  //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_owned_src_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_owned_src_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_owned_src_mutex_lock();
  #endif

  // 將本 src 從 ctx_list_owned_src 中刪除.
  std::deque<mws_src_t*>::iterator it = this->ctx_ptr->ctx_list_owned_src.begin();
  while (it != this->ctx_ptr->ctx_list_owned_src.end())
  {
    if (*it == this)
    {
      it = this->ctx_ptr->ctx_list_owned_src.erase(it);
      break; // while (it != this->ctx_ptr->ctx_list_owned_src.end())
    }
    else
    {
      if (it != this->ctx_ptr->ctx_list_owned_src.end())
      {
        ++it;
      }
    }
  }

  //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_owned_src_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_owned_src_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_owned_src_mutex_unlock();
  #endif

  std::string log_body = "mws_src(" + this->topic_name + ") destructor complete";
  write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

void mws_src::src_send_error(fd_t fd,
                             const std::string function,
                             const int line_no)
{
  if ((g_fd_table[fd].status == FD_STATUS_UNKNOWN) ||
      (g_fd_table[fd].status == FD_STATUS_SRC_CONN_FD_FAIL) ||
      (g_fd_table[fd].status == FD_STATUS_SRC_CONN_WAIT_TO_CLOSE))
  {
    std::string log_body_role_addr_info;
    fd_info_log(fd, log_body_role_addr_info);
    std::string log_body = function + " error. " + log_body_role_addr_info;
    write_to_log(this->topic_name, -1, "E", __FILE__, function, line_no, log_body);

    return;
  }

  // Begin: 移除 src 內 src_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].src_conn_ptr->erase_src_connect_fds(fd);
    if (rtv != 0)
    {
      // src_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in src_connect_fds";
      write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 src 內 src_connect_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].src_conn_ptr->ctx_ptr->erase_ctx_list_owned_src_conn_fds(fd);
    if (rtv != 0)
    {
      // src_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_owned_src_conn_fds";
      write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(fd, &g_fd_table[fd].src_conn_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove src conn fd:" +
                           std::to_string(fd) +
                           " from all_set ";
    write_to_log("", 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_SRC_CONN_FD_FAIL.
  //g_fd_table[fd].status = FD_STATUS_SRC_CONN_FD_FAIL;
  update_g_fd_table_status(fd,
                           FD_STATUS_SRC_CONN_FD_FAIL,
                           __func__,
                           __LINE__);

  // (在 dispatch MWS_SRC_EVENT_DISCONNECT 時放入) 把 fd 放入 ctx 的 ctx_list_wait_to_close_src_conn_fds.
  //g_fd_table[fd].src_conn_ptr->ctx_ptr->ctx_list_wait_to_close_src_conn_fds.push_back(fd);

  // 產生 MWS_SRC_EVENT_DISCONNECT.
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf create MWS_SRC_EVENT_DISCONNECT" << std::endl;
  mws_event_t* event_ptr = g_fd_table[fd].src_conn_ptr->evq_ptr->create_non_msg_event(fd, MWS_SRC_EVENT_DISCONNECT, false);
  g_fd_table[fd].src_conn_ptr->evq_ptr->push_back_non_msg_event(event_ptr);
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af create MWS_SRC_EVENT_DISCONNECT" << std::endl;

  // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
  if (g_mws_log_level >= 1)
  {
    std::string log_body_role_addr_info;
    fd_info_log(fd, log_body_role_addr_info);
    std::string log_body = "create MWS_SRC_EVENT_DISCONNECT event. " + log_body_role_addr_info;
    write_to_log(this->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }
  // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

  std::string log_body_role_addr_info;
  fd_info_log(fd, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(this->topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

int mws_src::mws_src_send(const char* msg,
                          size_t len,
                          int flags)
{
  return this->mws_src::mws_hf_src_send(msg,
                                        len,
                                        0,
                                        flags);
}

// ST (Successful Transmission) 表示 fd 成功送出資料而且沒有資料存放在 deque.
// BLK (Block) 表示 fd 有資料存在 deque 但沒發生底層錯誤.
// F (Failure) 表示 fd 底曾發生非 EAGAIN/EWOULDBLOCK/EINTR 的錯誤, 會斷線.
// ** 所有 fd 有一個發生上述其中一個狀況, 就表示有 "發生",
//    全部 fd 都沒發生上述其中一個狀況, 就表示 "沒有" 發生.
// ** MWS_SEND_MSG_FLUSH 模式, fd 發生 F 或 BLK 會斷線.
// ** 非 MWS_SEND_MSG_FLUSH 模式, fd 發生 F 會斷線, 發生 BLK.
// 0: 沒有 F + 沒有 BLK + 發生 ST (全部 fd 傳送完成)
// 1: 沒有 F + 沒有 BLK + 沒有 ST (沒有接收端)
// 2: 沒有 F + 發生 BLK + 發生 ST
// 3: 沒有 F + 發生 BLK + 沒有 ST
// 4: 發生 F + 沒有 BLK + 發生 ST
// 5: 發生 F + 沒有 BLK + 沒有 ST (全部 fd 斷線)
// 6: 發生 F + 發生 BLK + 發生 ST
// 7: 發生 F + 發生 BLK + 沒有 ST

int mws_src::mws_hf_src_send(const char* msg,
                             size_t len,
                             uint64_t seq_num,
                             int flags)
{
  if (len > MAX_MSG_SIZE)
  {
    return (-1);
  }

  std::string log_body;

  // 組合 package.
  mws_pkg_t pkg_obj;
  pkg_obj.head.seq_num = g_endianness_obj.host_to_network_uint64_t(seq_num);
  ssize_t pkg_obj_size = (signed)(g_size_of_mws_pkg_head + len);
  pkg_obj.head.msg_size = htons(len);
  pkg_obj.head.msg_type = MSG_TYPE_MSG;
  memcpy(&pkg_obj.body[0], msg, len);

  std::vector<fd_t> temp_fds = this->src_connect_fds;

  // 依每個 fd 傳送訊息.
  size_t num_of_src_conn_fd = temp_fds.size();
  for (size_t i = 0; i < num_of_src_conn_fd; ++i)
  {
    bool flag_can_send_msg = true;
    bool flag_put_msg_into_send_buffer = false;
    bool flag_send_buffer_pop_all = true;
    ssize_t cumulative_number_of_bytes_sent = 0;

    // 要檢查 fd 連線狀態, 先 fd_lock().
    #if (MWS_DEBUG == 1)
      g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[temp_fds[i]].fd_lock();
    #endif
    // 當此 fd 為連線狀態才能送出 message.
    if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      g_fd_table[temp_fds[i]].send_buffer_ptr->lock();

      // 走過所有的 send buffer, 將此 fd 未送出的訊息全部送出.
      for (ssize_t j = 0;
           j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
           ++j)
      {
        mws_send_buff_t* msg_ptr =
          (mws_send_buff_t*)g_fd_table[temp_fds[i]].send_buffer_ptr->at(g_fd_table[temp_fds[i]].send_buffer_number, j);

        // 將指標指向剩餘資料的起點.
        char* remaining_msg_ptr = (char*)(&msg_ptr->buff[msg_ptr->pos]);
        // 將資料送出.
        ssize_t rtv = mws_send_nonblock(temp_fds[i], remaining_msg_ptr, msg_ptr->remaining_msg_len, 0);
        // 有送出全部 message.
        if (rtv == msg_ptr->remaining_msg_len)
        {
          msg_ptr->pos += rtv;
          msg_ptr->remaining_msg_len -= rtv;
        }
        // 遇到連續 EAGAIN/EWOULDBLOCK/EINTR 次數超過 MAX_NUM_OF_RETRIES_SEND.
        else if ((rtv < msg_ptr->remaining_msg_len) && (rtv != (-1)))
        {
          msg_ptr->pos += rtv;
          msg_ptr->remaining_msg_len -= rtv;

          flag_can_send_msg = false;
          flag_put_msg_into_send_buffer = true;
          flag_send_buffer_pop_all = false;

          break; // break for (ssize_t j = 0;
                 //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                 //            ++j)
        }
        // 該連線有問題, 清除連線資訊.
        else
        {
          // Begin: src conn fd 失效處理.
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->src_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
          // End: src conn fd 失效處理.

          flag_can_send_msg = false;
          //flag_put_msg_into_send_buffer = false;
          //flag_send_buffer_pop_all = true;

          break; // break for (ssize_t j = 0;
                 //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                 //            ++j)
        }

        // Begin: 檢查是否要處理下一筆 send buffer 資料.
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_lock();
          #endif

          // 如果此 fd 已經斷線, 不用再檢查 send buffer deque, 離開 for loop.
          if (g_fd_table[temp_fds[i]].status != FD_STATUS_SRC_CONN_READY)
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif

            flag_can_send_msg = false;
            //flag_put_msg_into_send_buffer = false;
            flag_send_buffer_pop_all = true;

            break; // break for (ssize_t j = 0;
                   //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                   //            ++j)
          }
          else
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
        }
        // End: 檢查是否要處理下一筆 send buffer 資料.
      } // for (ssize_t j = 0;
        //      j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
        //      ++j)

      if (flag_can_send_msg == true)
      {
        // 要檢查 fd 連線狀態, 先 fd_lock().
        #if (MWS_DEBUG == 1)
          g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[temp_fds[i]].fd_lock();
        #endif
        // 當此 fd 為連線狀態才能送出 message.
        if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 送出本次的 package.
          cumulative_number_of_bytes_sent =
            mws_send_nonblock(temp_fds[i], (char*)&pkg_obj, pkg_obj_size, 0);
          if (cumulative_number_of_bytes_sent == pkg_obj_size)
          {
            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
            //continue; // for (size_t i = 0; i < num_of_src_conn_fd; ++i)
          }
          else if (cumulative_number_of_bytes_sent >= 0)
          {
            // 將 message 放入 send buffer.
            flag_put_msg_into_send_buffer = true;
            flag_send_buffer_pop_all = false;
          }
          // 該連線有問題, 清除連線資訊.
          //else if (cumulative_number_of_bytes_sent < 0)
          else
          {
            // Begin: src conn fd 失效處理.
            {
              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_lock();
              #endif

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
              this->src_send_error(temp_fds[i], __func__, __LINE__);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_unlock();
              #endif
            }
            // End: src conn fd 失效處理.

            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
          }
        } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
        else
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 尚未連線.
          log_body = "mws_hf_src_send() fd: " +
                     std::to_string(temp_fds[i]) +
                     " status != FD_STATUS_SRC_CONN_READY (" +
                     std::to_string(g_fd_table[temp_fds[i]].status) +
                     ")";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
        }  // else of if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
      }

      if (flag_put_msg_into_send_buffer == true)
      {
        // Begin: 將 message 放入 send buffer.
        {
          mws_send_buff_t send_buff_obj;
          memcpy(&send_buff_obj.buff[0],
                 ((char*)&pkg_obj + cumulative_number_of_bytes_sent),
                 (pkg_obj_size - cumulative_number_of_bytes_sent));
          send_buff_obj.pos = 0;
          send_buff_obj.remaining_msg_len = (pkg_obj_size - cumulative_number_of_bytes_sent);
          // 新增一筆資料 (send_buff_obj) 在 deque 的最後面.
          //g_fd_table[temp_fds[i]].send_buffer_ptr->lock();
          int rtv = g_fd_table[temp_fds[i]].send_buffer_ptr->push_back(g_fd_table[temp_fds[i]].send_buffer_number,
                                                                       &send_buff_obj);
          //g_fd_table[temp_fds[i]].send_buffer_ptr->unlock();
          if (rtv != 0)
          {
            // 如果 send buffer 出問題, 斷線處理.
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->src_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif

            flag_send_buffer_pop_all = true;
          }
        }
        // End: 將 message 放入 send buffer.
      }

      if (flag_send_buffer_pop_all == true)
      {
        g_fd_table[temp_fds[i]].send_buffer_ptr->pop_all(g_fd_table[temp_fds[i]].send_buffer_number);
      }
      else
      {
        g_fd_table[temp_fds[i]].send_buffer_ptr->pop_head_to_prev(g_fd_table[temp_fds[i]].send_buffer_number);
      }

      g_fd_table[temp_fds[i]].send_buffer_ptr->unlock();
    } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
    else
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      // 尚未連線.
      log_body = "mws_hf_src_send() fd: " +
                 std::to_string(temp_fds[i]) +
                 " status != FD_STATUS_SRC_CONN_READY (" +
                 std::to_string(g_fd_table[temp_fds[i]].status) +
                 ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  } // for (size_t i = 0; i < num_of_src_conn_fd; ++i)

  return 0;
}

// heartbeat 用 non-block 模式傳送.
// 在沒有發生 fd 異常, 沒傳出去的訊息放到 send buffer.
void mws_src::mws_src_send_heartbeat()
{
  std::string log_body;

  // 組合 mws_pkg_head_t.
  mws_pkg_head_t pkg_head;
  //pkg_head.seq_num = g_endianness_obj.host_to_network_uint64_t(0);
  pkg_head.seq_num = 0;
  ssize_t pkg_head_size = (signed)(g_size_of_mws_pkg_head);
  //pkg_head.msg_size = htons(0);
  pkg_head.msg_size = 0;
  pkg_head.msg_type = MSG_TYPE_HB;

  std::vector<fd_t> temp_fds = this->src_connect_fds;

  // 依每個 fd 傳送訊息.
  size_t num_of_src_conn_fd = temp_fds.size();
  for (size_t i = 0; i < num_of_src_conn_fd; ++i)
  {
    bool flag_can_send_msg = true;
    bool flag_put_msg_into_send_buffer = false;
    bool flag_send_buffer_pop_all = true;
    ssize_t cumulative_number_of_bytes_sent = 0;

    // 要檢查 fd 連線狀態, 先 fd_lock().
    #if (MWS_DEBUG == 1)
      g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[temp_fds[i]].fd_lock();
    #endif
    // 當此 fd 為連線狀態才能送出 message.
    if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      g_fd_table[temp_fds[i]].send_buffer_ptr->lock();

      // 走過所有的 send buffer, 將此 fd 未送出的訊息全部送出.
      for (ssize_t j = 0;
           j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
           ++j)
      {
        mws_send_buff_t* msg_ptr =
          (mws_send_buff_t*)g_fd_table[temp_fds[i]].send_buffer_ptr->at(g_fd_table[temp_fds[i]].send_buffer_number, j);

        // 將指標指向剩餘資料的起點.
        char* remaining_msg_ptr = (char*)(&msg_ptr->buff[msg_ptr->pos]);
        // 將資料送出.
        ssize_t rtv = mws_send_nonblock(temp_fds[i], remaining_msg_ptr, msg_ptr->remaining_msg_len, 0);
        // 有送出全部 message.
        if (rtv == msg_ptr->remaining_msg_len)
        {
          msg_ptr->pos += rtv;
          msg_ptr->remaining_msg_len -= rtv;
        }
        // 遇到連續 EAGAIN/EWOULDBLOCK/EINTR 次數超過 MAX_NUM_OF_RETRIES_SEND.
        else if ((rtv < msg_ptr->remaining_msg_len) && (rtv != (-1)))
        {
          msg_ptr->pos += rtv;
          msg_ptr->remaining_msg_len -= rtv;

          flag_can_send_msg = false;
          flag_put_msg_into_send_buffer = true;
          flag_send_buffer_pop_all = false;

          break; // break for (ssize_t j = 0;
                 //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                 //            ++j)
        }
        // 該連線有問題, 清除連線資訊.
        else
        {
          // Begin: src conn fd 失效處理.
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->src_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
          // End: src conn fd 失效處理.

          flag_can_send_msg = false;
          //flag_put_msg_into_send_buffer = false;
          //flag_send_buffer_pop_all = true;

          break; // break for (ssize_t j = 0;
                 //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                 //            ++j)
        }

        // Begin: 檢查是否要處理下一筆 send buffer 資料.
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_lock();
          #endif

          // 如果此 fd 已經斷線, 不用再檢查 send buffer deque, 離開 for loop.
          if (g_fd_table[temp_fds[i]].status != FD_STATUS_SRC_CONN_READY)
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif

            flag_can_send_msg = false;
            //flag_put_msg_into_send_buffer = false;
            flag_send_buffer_pop_all = true;

            break; // break for (ssize_t j = 0;
                   //            j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
                   //            ++j)
          }
          else
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
        }
        // End: 檢查是否要處理下一筆 send buffer 資料.
      } // for (ssize_t j = 0;
        //      j < g_fd_table[temp_fds[i]].send_buffer_ptr->size(g_fd_table[temp_fds[i]].send_buffer_number);
        //      ++j)

      if (flag_can_send_msg == true)
      {
        // 要檢查 fd 連線狀態, 先 fd_lock().
        #if (MWS_DEBUG == 1)
          g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[temp_fds[i]].fd_lock();
        #endif
        // 當此 fd 為連線狀態才能送出 heartbeat.
        if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 送出本次的 package.
          cumulative_number_of_bytes_sent =
            mws_send_nonblock(temp_fds[i], (char*)&pkg_head, pkg_head_size, 0);
          if (cumulative_number_of_bytes_sent == pkg_head_size)
          {
            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
            //continue; // for (size_t i = 0; i < num_of_src_conn_fd; ++i)
          }
          else if (cumulative_number_of_bytes_sent >= 0)
          {
            // 將 heartbeat message 放入 send buffer.
            flag_put_msg_into_send_buffer = true;
            flag_send_buffer_pop_all = false;
          }
          // 該連線有問題, 清除連線資訊.
          //else if (cumulative_number_of_bytes_sent < 0)
          else
          {
            // Begin: src conn fd 失效處理.
            {
              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_lock();
              #endif

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
              this->src_send_error(temp_fds[i], __func__, __LINE__);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_unlock();
              #endif
            }
            // End: src conn fd 失效處理.

            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
          }
        } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
        else
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 尚未連線.
          log_body = "mws_src_send_heartbeat() fd: " +
                     std::to_string(temp_fds[i]) +
                     " status != FD_STATUS_SRC_CONN_READY (" +
                     std::to_string(g_fd_table[temp_fds[i]].status) +
                     ")";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
        }  // else of if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
      }

      if (flag_put_msg_into_send_buffer == true)
      {
        // Begin: 將 heartbeat message 放入 send buffer.
        {
          mws_send_buff_t send_buff_obj;
          memcpy(&send_buff_obj.buff[0],
                 ((char*)&pkg_head + cumulative_number_of_bytes_sent),
                 (pkg_head_size - cumulative_number_of_bytes_sent));
          send_buff_obj.pos = 0;
          send_buff_obj.remaining_msg_len = (pkg_head_size - cumulative_number_of_bytes_sent);
          // 新增一筆資料 (send_buff_obj) 在 deque 的最後面.
          //g_fd_table[temp_fds[i]].send_buffer_ptr->lock();
          int rtv = g_fd_table[temp_fds[i]].send_buffer_ptr->push_back(g_fd_table[temp_fds[i]].send_buffer_number,
                                                                       &send_buff_obj);
          //g_fd_table[temp_fds[i]].send_buffer_ptr->unlock();
          if (rtv != 0)
          {
            // 如果 send buffer 出問題, 斷線處理.
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->src_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif

            flag_send_buffer_pop_all = true;
          }
        }
        // End: 將 heartbeat message 放入 send buffer.
      }

      if (flag_send_buffer_pop_all == true)
      {
        g_fd_table[temp_fds[i]].send_buffer_ptr->pop_all(g_fd_table[temp_fds[i]].send_buffer_number);
      }
      else
      {
        g_fd_table[temp_fds[i]].send_buffer_ptr->pop_head_to_prev(g_fd_table[temp_fds[i]].send_buffer_number);
      }

      g_fd_table[temp_fds[i]].send_buffer_ptr->unlock();
    } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_SRC_CONN_READY)
    else
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      // 尚未連線.
      log_body = "mws_src_send_heartbeat() fd: " +
                 std::to_string(temp_fds[i]) +
                 " status != FD_STATUS_SRC_CONN_READY (" +
                 std::to_string(g_fd_table[temp_fds[i]].status) +
                 ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  } // for (size_t i = 0; i < num_of_src_conn_fd; ++i)

  return;
}

std::string mws_src::mws_get_cfg_section()
{
  return this->cfg_section;
}

uint32_t mws_src::mws_get_object_status()
{
  return this->object_status;
}

bool mws_src::mws_is_hot_failover_recv_mode()
{
  return this->is_hot_failover_recv_mode;
}

std::string mws_src::get_topic_name()
{
  return this->topic_name;
}

int mws_src::erase_src_connect_fds(fd_t fd)
{
  for (unsigned int i = 0; i < this->src_connect_fds.size(); ++i)
  {
    if (this->src_connect_fds[i] == fd)
    {
      this->src_connect_fds.erase((this->src_connect_fds.begin() + i));
      return 0;
    }
  }

  return 1;
}

// source callback 範例.
// 功能: 處理 source 的 event (內容由 AP 的程式設計者依照業務撰寫).
// 回傳值 0: 表示成功.
// 回傳值 非0: 表示失敗, 會跳離 mws_evq::dispatch_events() 和 mws_evq::mws_event_dispatch().
// 參數 event_ptr: 指向放置 event 的 buffer 的指標.
// 參數 custom_data_ptr: 指向放置 custom data 的 buffer 的指標.
// 參數 custom_data_length: custom data 的長度.
// 備註: src 可供使用的事件種類如下:
//       MWS_SRC_EVENT_CONNECT, MWS_SRC_EVENT_DISCONNECT, MWS_SRC_DATA, MWS_SRC_UNRECOVERABLE_LOSS.
int src_callback_example(mws_event_t* event_ptr,
                         void* custom_data_ptr,
                         size_t custom_data_length)
{
  try
  {
    switch (event_ptr->event_type)
    {
      case MWS_SRC_DATA:
      {
        // 接收到 rcv 傳來的訊息時.

        break;
      }
      case MWS_SRC_EVENT_CONNECT:
      {
        // 與 rcv 連線完成時.

        break;
      }
      case MWS_SRC_EVENT_DISCONNECT:
      {
        // 與 rcv 斷線時.

        break;
      }
      case MWS_SRC_UNRECOVERABLE_LOSS:
      {
        // 發現 rcv 傳來的訊息跳號時 (訊息丟失).

        break;
      }
      default:
      {
        // 其他非預期狀況.

        break;
      }
    }
  }
  catch (...)
  {
    // exception handling.

    return (-1);
  }

  return 0;
}
