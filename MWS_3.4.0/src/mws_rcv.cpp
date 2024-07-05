//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_RCV_CPP 1

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

mws_rcv_attr::mws_rcv_attr(std::string cfg_section)
{
  this->cfg_section = cfg_section;

  // rcv set from default.
  this->topic_name = "";

  sess_addr_pair_t sess;
  //sess.listen_addr.IP = "127.0.0.1";
  //sess.listen_addr.low_port = 1;
  //sess.listen_addr.high_port = 65535;
  //sess.listen_addr.next_bind_port = sess.listen_addr.low_port;
  //sess.conn_addr.IP = "127.0.0.1";
  //sess.conn_addr.low_port = 1;
  //sess.conn_addr.high_port = 65535;
  //sess.conn_addr.next_bind_port = sess.conn_addr.low_port;

  this->is_hot_failover_recv_mode = false;

  uint16_t temp_listen_port_range_low = 0;
  uint16_t temp_listen_port_range_high = 0;
  uint16_t temp_connect_port_range_low = 0;
  uint16_t temp_connect_port_range_high = 0;

  std::map<std::string, std::string> my_cfg;
  std::string temp_topic_name("");
  std::string default_section = "default_receiver_config_value";
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
      this->topic_name = temp_topic_name;
    }

    name = "listen_ip";
    if (std::string(my_cfg[name]).size() > 0)
    {
      sess.listen_addr.IP = std::string(my_cfg[name]);
    }
    if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
    {
      temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
    }
    if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
    {
      temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
    }
    set_port_high_low(sess.listen_addr,
                      temp_listen_port_range_low,
                      temp_listen_port_range_high,
                      __FILE__,
                      __func__,
                      __LINE__);

    name = "connect_ip";
    if (std::string(my_cfg[name]).size() > 0)
    {
      sess.conn_addr.IP = std::string(my_cfg[name]);
    }
    if (std::string(my_cfg["connect_port_range_low"]).size() > 0)
    {
      temp_connect_port_range_low = (uint16_t)atoi(my_cfg["connect_port_range_low"].c_str());
    }
    if (std::string(my_cfg["connect_port_range_high"]).size() > 0)
    {
      temp_connect_port_range_high = (uint16_t)atoi(my_cfg["connect_port_range_high"].c_str());
    }
    set_port_high_low(sess.conn_addr,
                      temp_connect_port_range_low,
                      temp_connect_port_range_high,
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
    //std::cout << "RCV_DEFAULT is_hot_failover_recv_mode: " << this->is_hot_failover_recv_mode << std::endl;
  }

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
      this->topic_name = temp_topic_name;
    }

    for (int i = 1; i < 100; ++i)
    {
      std::stringstream ss;
      ss << i;
      if (i < 10)
      {
        name = "sess_addr_pair_0" + ss.str();
      }
      else
      {
        name = "sess_addr_pair_" + ss.str();
      }

      //std::cout << "name: " << name << std::endl;

      // 設定 session 的 listen address 及 connect address.
      // sess_addr_pair 格式: [IP_1:port_1-port_2;IP_2:port_3-port_4]
      //                      [IP_1:port_1;IP_2:port_2-port_3]
      //                      [IP_1:port_1-port_2]
      //                      [IP_1:port_1]
      //                      []
      if (!std::string(my_cfg[name]).empty())
      {
        std::string temp_addr = std::string(my_cfg[name]);
        if (temp_addr == "[]")
        {

        }
        else
        {
          if (temp_addr.find(";") == temp_addr.npos)
          {
            // [IP_1:port_1]
            if (temp_addr.find("-") == temp_addr.npos)
            {
              std::size_t pos1 = temp_addr.find("[");
              std::size_t pos2 = temp_addr.find(":");
              sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
              sess.listen_addr.low_port = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
              sess.listen_addr.high_port = sess.listen_addr.low_port;
            }
            // [IP_1:port_1-port_2]
            else
            {
              std::size_t pos1 = temp_addr.find("[");
              std::size_t pos2 = temp_addr.find(":");
              std::size_t pos3 = temp_addr.find("-");
              sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
              uint16_t port1 = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
              uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos3 + 1).c_str());
              if (port1 < port2)
              {
                sess.listen_addr.low_port = port1;
                sess.listen_addr.high_port = port2;
              }
              else
              {
                sess.listen_addr.low_port = port2;
                sess.listen_addr.high_port = port1;
              }
            }
          }
          else
          {
            // [IP_1:port_1;IP_2:port_2-port_3]
            if (temp_addr.find("-") > temp_addr.find(";"))
            {
              std::size_t pos1 = temp_addr.find("[");
              std::size_t pos2 = temp_addr.find(":");
              std::size_t pos3 = temp_addr.find(";");
              sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
              sess.listen_addr.low_port = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
              sess.listen_addr.high_port = sess.listen_addr.low_port;

              temp_addr = temp_addr.substr(pos3 + 1);
              pos1 = temp_addr.find(":");
              pos2 = temp_addr.find("-");
              sess.conn_addr.IP = temp_addr.substr(0, pos1);
              uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos1 + 1, pos2).c_str());
              uint16_t port3 = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
              if (port2 < port3)
              {
                sess.conn_addr.low_port = port2;
                sess.conn_addr.high_port = port3;
              }
              else
              {
                sess.conn_addr.low_port = port3;
                sess.conn_addr.high_port = port2;
              }
            }
            // [IP_1:port_1-port_2;IP_2:port_3-port_4]
            else
            {
              std::size_t pos1 = temp_addr.find("[");
              std::size_t pos2 = temp_addr.find(":");
              std::size_t pos3 = temp_addr.find("-");
              std::size_t pos4 = temp_addr.find(";");
              sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
              uint16_t port1 = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
              uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos3 + 1, pos4).c_str());
              if (port1 < port2)
              {
                sess.listen_addr.low_port = port1;
                sess.listen_addr.high_port = port2;
              }
              else
              {
                sess.listen_addr.low_port = port2;
                sess.listen_addr.high_port = port1;
              }

              temp_addr = temp_addr.substr(pos4 + 1);
              pos1 = temp_addr.find(":");
              pos2 = temp_addr.find("-");
              sess.conn_addr.IP = temp_addr.substr(0, pos1);
              uint16_t port3 = (uint16_t)atoi(temp_addr.substr(pos1 + 1, pos2).c_str());
              uint16_t port4 = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
              if (port3 < port4)
              {
                sess.conn_addr.low_port = port3;
                sess.conn_addr.high_port = port4;
              }
              else
              {
                sess.conn_addr.low_port = port4;
                sess.conn_addr.high_port = port3;
              }
            }
          }
        }

        // add session pair to rcv session list.
        this->rcv_session_list[this->num_of_rcv_sessions++] = sess;
      }
    }

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
    //std::cout << "RCV is_hot_failover_recv_mode: " << this->is_hot_failover_recv_mode << std::endl;
  }

  return;
}

mws_rcv_attr::~mws_rcv_attr()
{
  return;
}

void mws_rcv_attr::mws_modify_rcv_attr(std::string attr_name, std::string attr_value)
{
  if (attr_name == "topic_name")
  {
    if ((attr_value.size() == 0) || (attr_value.size() > 64))
    {
      return;
    }
    else
    {
      this->topic_name = attr_value;
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

  if (attr_name.substr(0, 15) == "sess_addr_pair_")
  {
    unsigned int index = 0;

    if (attr_name.substr(15, 1) == "0")
    {
      index = (uint16_t)atoi(attr_name.substr(16, 1).c_str()) - 1;
      //std::cout << index << std::endl;
    }
    else
    {
      index = (uint16_t)atoi(attr_name.substr(15, 2).c_str()) - 1;
      //std::cout << index << std::endl;
    }

    if (index < this->num_of_rcv_sessions)
    {
      // 設定 session 的 listen address 及 connect address.
      // sess_addr_pair 格式: [IP_1:port_1-port_2;IP_2:port_3-port_4]
      //                      [IP_1:port_1;IP_2:port_2-port_3]
      //                      [IP_1:port_1-port_2]
      //                      [IP_1:port_1]
      if (!attr_value.empty())
      {
        if (attr_value.find(";") == attr_value.npos)
        {
          // [IP_1:port_1]
          if (attr_value.find("-") == attr_value.npos)
          {
            std::size_t pos1 = attr_value.find("[");
            std::size_t pos2 = attr_value.find(":");
            this->rcv_session_list[index].listen_addr.IP = attr_value.substr(pos1 + 1, pos2 - pos1 - 1);
            this->rcv_session_list[index].listen_addr.low_port = (uint16_t)atoi(attr_value.substr(pos2 + 1).c_str());
            this->rcv_session_list[index].listen_addr.high_port = this->rcv_session_list[index].listen_addr.low_port;
          }
          // [IP_1:port_1-port_2]
          else
          {
            std::size_t pos1 = attr_value.find("[");
            std::size_t pos2 = attr_value.find(":");
            std::size_t pos3 = attr_value.find("-");
            this->rcv_session_list[index].listen_addr.IP = attr_value.substr(pos1 + 1, pos2 - pos1 - 1);
            uint16_t port1 = (uint16_t)atoi(attr_value.substr(pos2 + 1, pos3).c_str());
            uint16_t port2 = (uint16_t)atoi(attr_value.substr(pos3 + 1).c_str());
            if (port1 < port2)
            {
              this->rcv_session_list[index].listen_addr.low_port = port1;
              this->rcv_session_list[index].listen_addr.high_port = port2;
            }
            else
            {
              this->rcv_session_list[index].listen_addr.low_port = port2;
              this->rcv_session_list[index].listen_addr.high_port = port1;
            }
          }
        }
        else
        {
          // [IP_1:port_1;IP_2:port_2-port_3]
          if (attr_value.find("-") > attr_value.find(";"))
          {
            std::size_t pos1 = attr_value.find("[");
            std::size_t pos2 = attr_value.find(":");
            std::size_t pos3 = attr_value.find(";");
            this->rcv_session_list[index].listen_addr.IP = attr_value.substr(pos1 + 1, pos2 - pos1 - 1);
            this->rcv_session_list[index].listen_addr.low_port = (uint16_t)atoi(attr_value.substr(pos2 + 1, pos3).c_str());
            this->rcv_session_list[index].listen_addr.high_port = this->rcv_session_list[index].listen_addr.low_port;

            attr_value = attr_value.substr(pos3 + 1);
            pos1 = attr_value.find(":");
            pos2 = attr_value.find("-");
            this->rcv_session_list[index].conn_addr.IP = attr_value.substr(0, pos1);
            uint16_t port2 = (uint16_t)atoi(attr_value.substr(pos1 + 1, pos2).c_str());
            uint16_t port3 = (uint16_t)atoi(attr_value.substr(pos2 + 1).c_str());
            if (port2 < port3)
            {
              this->rcv_session_list[index].conn_addr.low_port = port2;
              this->rcv_session_list[index].conn_addr.high_port = port3;
            }
            else
            {
              this->rcv_session_list[index].conn_addr.low_port = port3;
              this->rcv_session_list[index].conn_addr.high_port = port2;
            }
          }
          // [IP_1:port_1-port_2;IP_2:port_3-port_4]
          else
          {
            std::size_t pos1 = attr_value.find("[");
            std::size_t pos2 = attr_value.find(":");
            std::size_t pos3 = attr_value.find("-");
            std::size_t pos4 = attr_value.find(";");
            this->rcv_session_list[index].listen_addr.IP = attr_value.substr(pos1 + 1, pos2 - pos1 - 1);
            uint16_t port1 = (uint16_t)atoi(attr_value.substr(pos2 + 1, pos3).c_str());
            uint16_t port2 = (uint16_t)atoi(attr_value.substr(pos3 + 1, pos4).c_str());
            if (port1 < port2)
            {
              this->rcv_session_list[index].listen_addr.low_port = port1;
              this->rcv_session_list[index].listen_addr.high_port = port2;
            }
            else
            {
              this->rcv_session_list[index].listen_addr.low_port = port2;
              this->rcv_session_list[index].listen_addr.high_port = port1;
            }

            attr_value = attr_value.substr(pos4 + 1);
            pos1 = attr_value.find(":");
            pos2 = attr_value.find("-");
            this->rcv_session_list[index].conn_addr.IP = attr_value.substr(0, pos1);
            uint16_t port3 = (uint16_t)atoi(attr_value.substr(pos1 + 1, pos2).c_str());
            uint16_t port4 = (uint16_t)atoi(attr_value.substr(pos2 + 1).c_str());
            if (port3 < port4)
            {
              this->rcv_session_list[index].conn_addr.low_port = port3;
              this->rcv_session_list[index].conn_addr.high_port = port4;
            }
            else
            {
              this->rcv_session_list[index].conn_addr.low_port = port4;
              this->rcv_session_list[index].conn_addr.high_port = port3;
            }
          }
        }
      }
    }
    //else
    //{
      //std::cout << "no session to modify" << std::endl;
    //}
  }

  return;
}

int32_t mws_init_rcv(mws_rcv* rcv_ptr,
                     mws_ctx_t* ctx_ptr,
                     mws_evq_t* evq_ptr,
                     callback_t* rcv_cb_ptr,
                     void* custom_data_ptr,
                     const size_t custom_data_size,
                     const bool is_from_cfg,
                     const mws_rcv_attr_t mws_rcv_attr,
                     const std::string cfg_section)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  std::string log_body;

  rcv_ptr->object_status = 0;

  if (is_from_cfg == false)
  {
    rcv_ptr->cfg_section = mws_rcv_attr.cfg_section;
    // topic name.
    if ((mws_rcv_attr.topic_name.size() == 0) ||
        (mws_rcv_attr.topic_name.size() > 64))
    {
      log_body = "rcv topic size error or no topic name, topic name: ";
      log_body += mws_rcv_attr.topic_name;
      log_body += ", size: ";
      log_body += std::to_string(mws_rcv_attr.topic_name.size());
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      rcv_ptr->object_status = MWS_ERROR_TOPIC_NAME;

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
      rcv_ptr->topic_name = mws_rcv_attr.topic_name;
    }
    // 接收資料模式.
    rcv_ptr->is_hot_failover_recv_mode = mws_rcv_attr.is_hot_failover_recv_mode;

    rcv_ptr->num_of_rcv_sessions = mws_rcv_attr.num_of_rcv_sessions;
    for (size_t i = 0; i < rcv_ptr->num_of_rcv_sessions; ++i)
    {
      rcv_ptr->rcv_session_list[i] = mws_rcv_attr.rcv_session_list[i];
    }
  } // if (is_from_cfg == false)
  else
  {
    rcv_ptr->cfg_section = cfg_section;

    rcv_ptr->num_of_rcv_sessions = 0;

    // Begin: rcv set from default.
    rcv_ptr->topic_name = "";

    sess_addr_pair_t sess;
    //sess.listen_addr.IP = "127.0.0.1";
    //sess.listen_addr.low_port = 1;
    //sess.listen_addr.high_port = 65535;
    //sess.listen_addr.next_bind_port = sess.listen_addr.low_port;
    //sess.conn_addr.IP = "127.0.0.1";
    //sess.conn_addr.low_port = 1;
    //sess.conn_addr.high_port = 65535;
    //sess.conn_addr.next_bind_port = sess.conn_addr.low_port;

    rcv_ptr->is_hot_failover_recv_mode = false;

    uint16_t temp_listen_port_range_low = 0;
    uint16_t temp_listen_port_range_high = 0;
    uint16_t temp_connect_port_range_low = 0;
    uint16_t temp_connect_port_range_high = 0;
    // End: rcv set from default.
    // Begin: 從 cfg 的 default 取得設定值.
    std::map<std::string, std::string> my_cfg;
    std::string temp_topic_name("");
    std::string default_section = "default_receiver_config_value";
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
        rcv_ptr->topic_name = temp_topic_name;
      }

      name = "listen_ip";
      if (std::string(my_cfg[name]).size() > 0)
      {
        sess.listen_addr.IP = std::string(my_cfg[name]);
      }
      if (std::string(my_cfg["listen_port_range_low"]).size() > 0)
      {
        temp_listen_port_range_low = (uint16_t)atoi(my_cfg["listen_port_range_low"].c_str());
      }
      if (std::string(my_cfg["listen_port_range_high"]).size() > 0)
      {
        temp_listen_port_range_high = (uint16_t)atoi(my_cfg["listen_port_range_high"].c_str());
      }
      set_port_high_low(sess.listen_addr,
                        temp_listen_port_range_low,
                        temp_listen_port_range_high,
                        __FILE__,
                        __func__,
                        __LINE__);

      name = "connect_ip";
      if (std::string(my_cfg[name]).size() > 0)
      {
        sess.conn_addr.IP = std::string(my_cfg[name]);
      }
      if (std::string(my_cfg["connect_port_range_low"]).size() > 0)
      {
        temp_connect_port_range_low = (uint16_t)atoi(my_cfg["connect_port_range_low"].c_str());
      }
      if (std::string(my_cfg["connect_port_range_high"]).size() > 0)
      {
        temp_connect_port_range_high = (uint16_t)atoi(my_cfg["connect_port_range_high"].c_str());
      }
      set_port_high_low(sess.conn_addr,
                        temp_connect_port_range_low,
                        temp_connect_port_range_high,
                        __FILE__,
                        __func__,
                        __LINE__);

      // 設定是否為 hot failover recv.
      name = "is_hot_failover_recv_mode";
      if (my_cfg[name] == "Y")
      {
        rcv_ptr->is_hot_failover_recv_mode = true;
      }
      if (my_cfg[name] == "N")
      {
        rcv_ptr->is_hot_failover_recv_mode = false;
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
        rcv_ptr->topic_name = temp_topic_name;
      }

      if ((rcv_ptr->topic_name.size() == 0) ||
          (rcv_ptr->topic_name.size() > 64))
      {
        log_body = "rcv topic size error or no topic name, topic name: ";
        log_body += temp_topic_name;
        log_body += ", size: ";
        log_body += std::to_string(temp_topic_name.size());
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        rcv_ptr->object_status = MWS_ERROR_TOPIC_NAME;

        //pthread_mutex_unlock(&g_mws_global_mutex);
        #if (MWS_DEBUG == 1)
          g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_mws_global_mutex_unlock();
        #endif

        return -1;
      }

      for (int i = 1; i < 100; ++i)
      {
        std::stringstream ss;
        ss << i;
        if (i < 10)
        {
          name = "sess_addr_pair_0" + ss.str();
        }
        else
        {
          name = "sess_addr_pair_" + ss.str();
        }

        // 設定 session 的 listen address 及 connect address.
        // sess_addr_pair 格式: [IP_1:port_1-port_2;IP_2:port_3-port_4]
        //                      [IP_1:port_1;IP_2:port_2-port_3]
        //                      [IP_1:port_1-port_2]
        //                      [IP_1:port_1]
        //                      []
        if (!std::string(my_cfg[name]).empty())
        {
          std::string temp_addr = std::string(my_cfg[name]);
          if (temp_addr == "[]")
          {

          }
          else
          {
            if (temp_addr.find(";") == temp_addr.npos)
            {
              // [IP_1:port_1]
              if (temp_addr.find("-") == temp_addr.npos)
              {
                std::size_t pos1 = temp_addr.find("[");
                std::size_t pos2 = temp_addr.find(":");
                sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
                sess.listen_addr.low_port = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
                sess.listen_addr.high_port = sess.listen_addr.low_port;
              }
              // [IP_1:port_1-port_2]
              else
              {
                std::size_t pos1 = temp_addr.find("[");
                std::size_t pos2 = temp_addr.find(":");
                std::size_t pos3 = temp_addr.find("-");
                sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
                uint16_t port1 = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
                uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos3 + 1).c_str());
                if (port1 < port2)
                {
                  sess.listen_addr.low_port = port1;
                  sess.listen_addr.high_port = port2;
                }
                else
                {
                  sess.listen_addr.low_port = port2;
                  sess.listen_addr.high_port = port1;
                }
              }
            }
            else
            {
              // [IP_1:port_1;IP_2:port_2-port_3]
              if (temp_addr.find("-") > temp_addr.find(";"))
              {
                std::size_t pos1 = temp_addr.find("[");
                std::size_t pos2 = temp_addr.find(":");
                std::size_t pos3 = temp_addr.find(";");
                sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
                sess.listen_addr.low_port = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
                sess.listen_addr.high_port = sess.listen_addr.low_port;

                temp_addr = temp_addr.substr(pos3 + 1);
                pos1 = temp_addr.find(":");
                pos2 = temp_addr.find("-");
                sess.conn_addr.IP = temp_addr.substr(0, pos1);
                uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos1 + 1, pos2).c_str());
                uint16_t port3 = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
                if (port2 < port3)
                {
                  sess.conn_addr.low_port = port2;
                  sess.conn_addr.high_port = port3;
                }
                else
                {
                  sess.conn_addr.low_port = port3;
                  sess.conn_addr.high_port = port2;
                }
              }
              // [IP_1:port_1-port_2;IP_2:port_3-port_4]
              else
              {
                std::size_t pos1 = temp_addr.find("[");
                std::size_t pos2 = temp_addr.find(":");
                std::size_t pos3 = temp_addr.find("-");
                std::size_t pos4 = temp_addr.find(";");
                sess.listen_addr.IP = temp_addr.substr(pos1 + 1, pos2 - pos1 - 1);
                //temp_listen_addr.low_port = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
                //temp_listen_addr.high_port = (uint16_t)atoi(temp_addr.substr(pos3 + 1, pos4).c_str());
                uint16_t port1 = (uint16_t)atoi(temp_addr.substr(pos2 + 1, pos3).c_str());
                uint16_t port2 = (uint16_t)atoi(temp_addr.substr(pos3 + 1, pos4).c_str());
                if (port1 < port2)
                {
                  sess.listen_addr.low_port = port1;
                  sess.listen_addr.high_port = port2;
                }
                else
                {
                  sess.listen_addr.low_port = port2;
                  sess.listen_addr.high_port = port1;
                }

                temp_addr = temp_addr.substr(pos4 + 1);
                pos1 = temp_addr.find(":");
                pos2 = temp_addr.find("-");
                sess.conn_addr.IP = temp_addr.substr(0, pos1);
                uint16_t port3 = (uint16_t)atoi(temp_addr.substr(pos1 + 1, pos2).c_str());
                uint16_t port4 = (uint16_t)atoi(temp_addr.substr(pos2 + 1).c_str());
                if (port3 < port4)
                {
                  sess.conn_addr.low_port = port3;
                  sess.conn_addr.high_port = port4;
                }
                else
                {
                  sess.conn_addr.low_port = port4;
                  sess.conn_addr.high_port = port3;
                }
              }
            }
          }

          // add session pair to rcv session list.
          rcv_ptr->rcv_session_list[rcv_ptr->num_of_rcv_sessions++] = sess;
        }
      }

      // 設定是否為 hot failover recv.
      name = "is_hot_failover_recv_mode";
      if (my_cfg[name] == "Y")
      {
        rcv_ptr->is_hot_failover_recv_mode = true;
      }
      if (my_cfg[name] == "N")
      {
        rcv_ptr->is_hot_failover_recv_mode = false;
      }
    }
    // End: 從設定的 cfg section 取得設定值.
  } // else of if (is_from_cfg == false)

  rcv_ptr->ctx_ptr = ctx_ptr;
  rcv_ptr->evq_ptr = evq_ptr;
  rcv_ptr->max_seq_num = 0;
  rcv_ptr->cb_ptr = rcv_cb_ptr;

  // 設定帶入 cb 的資料及資料長度.
  rcv_ptr->custom_data_size = custom_data_size;
  rcv_ptr->custom_data_ptr = calloc(1, rcv_ptr->custom_data_size);
  memcpy(rcv_ptr->custom_data_ptr, custom_data_ptr, rcv_ptr->custom_data_size);

  for (size_t i = 0; i < rcv_ptr->num_of_rcv_sessions; ++i)
  {
    // Begin: 將 rcv 要做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
    {
      wait_to_connect_rcv_session_t temp_obj;
      temp_obj.rcv_ptr = rcv_ptr;
      temp_obj.rcv_connection_setting = rcv_ptr->rcv_session_list[i];
      temp_obj.next_port = rcv_ptr->rcv_session_list[i].listen_addr.low_port;
      temp_obj.try_cnt = 0;
      rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
    }
    // End: 將 rcv 要做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  }

  rcv_ptr->flag_ready_to_release_rcv = false;

  //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_rcv_mutex));
  #if (MWS_DEBUG == 1)
    ctx_ptr->ctx_list_owned_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    ctx_ptr->ctx_list_owned_rcv_mutex_lock();
  #endif

  rcv_ptr->ctx_ptr->ctx_list_owned_rcv.push_back(rcv_ptr);

  //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_rcv_mutex));
  #if (MWS_DEBUG == 1)
    ctx_ptr->ctx_list_owned_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    ctx_ptr->ctx_list_owned_rcv_mutex_unlock();
  #endif

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return 0;
}

mws_rcv::mws_rcv(mws_rcv_attr_t mws_rcv_attr,
                 mws_ctx_t* ctx_ptr,
                 mws_evq_t* evq_ptr,
                 callback_t* rcv_cb_ptr,
                 void* custom_data_ptr,
                 const size_t custom_data_size)
{
  // 無用但需要的變數.
  std::string cfg_section("");
  int32_t rtv = mws_init_rcv(this,
                             ctx_ptr,
                             evq_ptr,
                             rcv_cb_ptr,
                             custom_data_ptr,
                             custom_data_size,
                             false,
                             mws_rcv_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_rcv(" + this->topic_name + ") constructor complete";
    write_to_log(this->topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_rcv(" + this->topic_name + ") constructor fail";
    write_to_log(this->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_rcv::mws_rcv(std::string cfg_section,
                 mws_ctx_t* ctx_ptr,
                 mws_evq_t* evq_ptr,
                 callback_t* rcv_cb_ptr,
                 void* custom_data_ptr,
                 const size_t custom_data_size)
{
  // 無用但需要的變數.
  mws_rcv_attr_t mws_rcv_attr("");
  int32_t rtv = mws_init_rcv(this,
                             ctx_ptr,
                             evq_ptr,
                             rcv_cb_ptr,
                             custom_data_ptr,
                             custom_data_size,
                             true,
                             mws_rcv_attr,
                             cfg_section);
  if (rtv == 0)
  {
    std::string log_body;
    log_body = "mws_rcv(" + this->topic_name + ") constructor complete";
    write_to_log(this->topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }
  else
  {
    std::string log_body;
    log_body = "mws_rcv(" + this->topic_name + ") constructor fail";
    write_to_log(this->topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

mws_rcv::~mws_rcv()
{
  //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock();
  #endif
  this->ctx_ptr->ctx_list_wait_to_stop_rcv.push_back(this);
  while (this->flag_ready_to_release_rcv == false)
  {
    //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock();
    #endif
    sleep(1);
    //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
    #if (MWS_DEBUG == 1)
      this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_lock();
    #endif
  }
  //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_wait_to_stop_rcv_mutex_unlock();
  #endif

  //pthread_mutex_lock(&(this->ctx_ptr->ctx_list_owned_rcv_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_owned_rcv_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_owned_rcv_mutex_lock();
  #endif

  // 將本 rcv 從 ctx_list_owned_rcv 中刪除.
  std::deque<mws_rcv_t*>::iterator it = this->ctx_ptr->ctx_list_owned_rcv.begin();
  while (it != this->ctx_ptr->ctx_list_owned_rcv.end())
  {
    if (*it == this)
    {
      it = this->ctx_ptr->ctx_list_owned_rcv.erase(it);
      break; // while (it != this->ctx_ptr->ctx_list_owned_rcv.end())
    }
    else
    {
      if (it != this->ctx_ptr->ctx_list_owned_rcv.end())
      {
        ++it;
      }
    }
  }

  //pthread_mutex_unlock(&(this->ctx_ptr->ctx_list_owned_rcv_mutex));
  #if (MWS_DEBUG == 1)
    this->ctx_ptr->ctx_list_owned_rcv_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    this->ctx_ptr->ctx_list_owned_rcv_mutex_unlock();
  #endif

  std::string log_body = "mws_rcv(" + this->topic_name + ") destructor complete";
  write_to_log(this->topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);

  return;
}

void mws_rcv::rcv_send_error(fd_t fd,
                             const std::string function,
                             const int line_no)
{
  if ((g_fd_table[fd].status == FD_STATUS_UNKNOWN) ||
      (g_fd_table[fd].status == FD_STATUS_RCV_FD_FAIL) ||
      (g_fd_table[fd].status == FD_STATUS_RCV_WAIT_TO_CLOSE))
  {
    std::string log_body_role_addr_info;
    fd_info_log(fd, log_body_role_addr_info);
    std::string log_body = function + " error. " + log_body_role_addr_info;
    write_to_log(this->topic_name, -1, "E", __FILE__, function, line_no, log_body);

    return;
  }

  // Begin: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].rcv_ptr->erase_rcv_list_connected_src_address(g_fd_table[fd].rcv_listen_addr_info);
    if (rtv != 0)
    {
      str_ip_port_t rcv_listen_addr;
      sockaddr_in_t_to_string(g_fd_table[fd].rcv_listen_addr_info,
                              rcv_listen_addr.str_ip,
                              rcv_listen_addr.str_port);

      // rcv_list_connected_src_address 沒有該 sockaddr_in_t 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) +
                 "(" + rcv_listen_addr.str_ip + ":" + rcv_listen_addr.str_port +
                 ") does not exist in rcv_listen_addr_info";
      write_to_log(this->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.

  // Begin: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].rcv_ptr->erase_rcv_connect_fds(fd);
    if (rtv != 0)
    {
      // rcv_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in rcv_connect_fds";
      write_to_log(this->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].rcv_ptr->ctx_ptr->erase_ctx_list_owned_rcv_fds(fd);
    if (rtv != 0)
    {
      // rcv_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_owned_rcv_fds";
      write_to_log(this->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(fd, &g_fd_table[fd].rcv_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove rcv fd:" +
                           std::to_string(fd) +
                           " from all_set ";
    write_to_log(this->topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_FD_FAIL.
  update_g_fd_table_status(fd,
                           FD_STATUS_RCV_FD_FAIL,
                           __func__,
                           __LINE__);

  // (在 dispatch MWS_MSG_EOS 時放入) 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
  //g_fd_table[fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(fd);

  // Begin: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  {
    wait_to_connect_rcv_session_t temp_obj;
    temp_obj.rcv_ptr = g_fd_table[fd].rcv_ptr;
    temp_obj.rcv_connection_setting = g_fd_table[fd].rcv_connection_setting;
    temp_obj.next_port = g_fd_table[fd].rcv_connection_setting.listen_addr.low_port;
    temp_obj.try_cnt = 0;
    g_fd_table[fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
  }
  // End: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.

  // 產生 MWS_MSG_EOS.
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf create MWS_MSG_EOS" << std::endl;
  mws_event_t* event_ptr = g_fd_table[fd].rcv_ptr->evq_ptr->create_non_msg_event(fd, MWS_MSG_EOS, false);
  g_fd_table[fd].rcv_ptr->evq_ptr->push_back_non_msg_event(event_ptr);
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " create MWS_MSG_EOS" << std::endl;

  // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
  if (g_mws_log_level >= 1)
  {
    std::string log_body_role_addr_info;
    fd_info_log(fd, log_body_role_addr_info);
    std::string log_body = "create MWS_MSG_EOS event. " + log_body_role_addr_info;
    write_to_log(this->topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }
  // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

  std::string log_body_role_addr_info;
  fd_info_log(fd, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(this->topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

int mws_rcv::mws_rcv_send(const char* msg,
                          size_t len,
                          int flags)
{
  return this->mws_rcv::mws_hf_rcv_send(msg,
                                        len,
                                        0,
                                        flags);
}

int mws_rcv::mws_hf_rcv_send(const char* msg,
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

  std::vector<fd_t> temp_fds = this->rcv_connect_fds;

  // 依每個 fd 傳送訊息.
  size_t num_of_rcv_conn_fd = temp_fds.size();
  for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)
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
    if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
          // Begin: rcv conn fd 失效處理.
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->rcv_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
          // End: rcv conn fd 失效處理.

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
          if (g_fd_table[temp_fds[i]].status != FD_STATUS_RCV_READY)
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
        if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
            //continue; // for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)
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
            // Begin: rcv conn fd 失效處理.
            {
              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_lock();
              #endif

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
              this->rcv_send_error(temp_fds[i], __func__, __LINE__);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_unlock();
              #endif
            }
            // End: rcv conn fd 失效處理.

            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
          }
        } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
        else
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 尚未連線.
          log_body = "mws_hf_rcv_send() fd: " +
                     std::to_string(temp_fds[i]) +
                     " status != FD_STATUS_RCV_READY (" +
                     std::to_string(g_fd_table[temp_fds[i]].status) +
                     ")";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
        }  // else of if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
            this->rcv_send_error(temp_fds[i], __func__, __LINE__);
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
    } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
    else
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      // 尚未連線.
      log_body = "mws_hf_rcv_send() fd: " +
                 std::to_string(temp_fds[i]) +
                 " status != FD_STATUS_RCV_READY (" +
                 std::to_string(g_fd_table[temp_fds[i]].status) +
                 ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  } // for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)

  return 0;
}

// heartbeat 用 non-block 模式傳送.
// 在沒有發生 fd 異常, 沒傳出去的訊息放到 send buffer.
void mws_rcv::mws_rcv_send_heartbeat()
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

  std::vector<fd_t> temp_fds = this->rcv_connect_fds;

  // 依每個 fd 傳送訊息.
  size_t num_of_rcv_conn_fd = temp_fds.size();
  for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)
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
    if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
          // Begin: rcv conn fd 失效處理.
          {
            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_lock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
            this->rcv_send_error(temp_fds[i], __func__, __LINE__);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            #if (MWS_DEBUG == 1)
              g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_fd_table[temp_fds[i]].fd_unlock();
            #endif
          }
          // End: rcv conn fd 失效處理.

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
          if (g_fd_table[temp_fds[i]].status != FD_STATUS_RCV_READY)
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
        if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
            //continue; // for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)
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
            // Begin: rcv conn fd 失效處理.
            {
              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_lock();
              #endif

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
              this->rcv_send_error(temp_fds[i], __func__, __LINE__);
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

              #if (MWS_DEBUG == 1)
                g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
              #else
                g_fd_table[temp_fds[i]].fd_unlock();
              #endif
            }
            // End: rcv conn fd 失效處理.

            //flag_put_msg_into_send_buffer = false;
            //flag_send_buffer_pop_all = true;
          }
        } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
        else
        {
          #if (MWS_DEBUG == 1)
            g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[temp_fds[i]].fd_unlock();
          #endif

          // 尚未連線.
          log_body = "mws_rcv_send_heartbeat() fd: " +
                     std::to_string(temp_fds[i]) +
                     " status != FD_STATUS_RCV_READY (" +
                     std::to_string(g_fd_table[temp_fds[i]].status) +
                     ")";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
        }  // else of if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
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
            this->rcv_send_error(temp_fds[i], __func__, __LINE__);
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
    } // if (g_fd_table[temp_fds[i]].status == FD_STATUS_RCV_READY)
    else
    {
      #if (MWS_DEBUG == 1)
        g_fd_table[temp_fds[i]].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[temp_fds[i]].fd_unlock();
      #endif

      // 尚未連線.
      log_body = "mws_rcv_send_heartbeat() fd: " +
                 std::to_string(temp_fds[i]) +
                 " status != FD_STATUS_RCV_READY (" +
                 std::to_string(g_fd_table[temp_fds[i]].status) +
                 ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    }
  } // for (size_t i = 0; i < num_of_rcv_conn_fd; ++i)

  return;
}

std::string mws_rcv::mws_get_cfg_section()
{
  return this->cfg_section;
}

uint32_t mws_rcv::mws_get_object_status()
{
  return this->object_status;
}

bool mws_rcv::mws_is_hot_failover_recv_mode()
{
  return this->is_hot_failover_recv_mode;
}

size_t mws_rcv::mws_get_num_of_rcv_sessions()
{
  return this->num_of_rcv_sessions;
}

std::string mws_rcv::get_topic_name()
{
  return this->topic_name;
}

int mws_rcv::erase_rcv_connect_fds(fd_t fd)
{
  for (unsigned int i = 0; i < this->rcv_connect_fds.size(); ++i)
  {
    if (this->rcv_connect_fds[i] == fd)
    {
      this->rcv_connect_fds.erase((this->rcv_connect_fds.begin() + i));
      return 0;
    }
  }

  return 1;
}

int mws_rcv::erase_rcv_list_connected_src_address(sockaddr_in_t addr_info)
{
  for (std::deque<sockaddr_in_t>::iterator it = this->rcv_list_connected_src_address.begin();
       it != this->rcv_list_connected_src_address.end();
       ++it)
  {
    if ((it->sin_port == addr_info.sin_port) &&
        (it->sin_addr.s_addr == addr_info.sin_addr.s_addr))
    {
      this->rcv_list_connected_src_address.erase(it);
      return 0;
    }
  }

  return 1;
}

// receiver callback 範例.
// 功能: 處理 receiver 的 event (內容由 AP 的程式設計者依照業務撰寫).
// 回傳值 0: 表示成功.
// 回傳值 非0: 表示失敗, 會跳離 mws_evq::dispatch_events() 和 mws_evq::mws_event_dispatch().
// 參數 event_ptr: 指向放置 event 的 buffer 的指標.
// 參數 custom_data_ptr: 指向放置 custom data 的 buffer 的指標.
// 參數 custom_data_length: custom data 的長度.
// 備註: rcv 可供使用的事件種類如下:
//       MWS_MSG_BOS, MWS_MSG_EOS, MWS_MSG_DATA, MWS_MSG_UNRECOVERABLE_LOSS.
int rcv_callback_example(mws_event_t* event_ptr,
                         void* custom_data_ptr,
                         size_t custom_data_length)
{
/*
  try
  {
    switch (event_ptr->event_type)
    {
      case MWS_MSG_DATA:
      {
        // 接收到 src 傳來的訊息時.

        break;
      }
      case MWS_MSG_BOS:
      {
        // 與 src 連線完成時.

        break;
      }
      case MWS_MSG_EOS:
      {
        // 與 src 斷線時.

        break;
      }
      case MWS_MSG_UNRECOVERABLE_LOSS:
      {
        // 發現 src 傳來的訊息跳號時 (訊息丟失).

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
*/
  return 0;
}
