//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_TYPE_DEFINITION_CPP 1

#include <ctime>
#include <iostream>
#include <string>
#include <strings.h>

#include "../inc/mws_global_variable.h"
#include "../inc/mws_type_definition.h"

using namespace mws_global_variable;

bool ip_port::operator ==(const ip_port& addr) const
{
  if ((this->IP == addr.IP) &&
      (this->port == addr.port))
  {
    return true;
  }

  return false;
}

sess_addr_pair::sess_addr_pair()
{
  this->listen_addr.IP = "127.0.0.1";
  this->listen_addr.low_port = 1;
  this->listen_addr.high_port = 65535;
  this->listen_addr.next_bind_port = this->listen_addr.low_port;

  this->conn_addr.IP = "127.0.0.1";
  this->conn_addr.low_port = 1;
  this->conn_addr.high_port = 65535;
  this->conn_addr.next_bind_port = this->conn_addr.low_port;

  return;
}

void sess_addr_pair::operator =(const sess_addr_pair& addr_pair)
{
  this->listen_addr.IP = addr_pair.listen_addr.IP;
  this->listen_addr.low_port = addr_pair.listen_addr.low_port;
  this->listen_addr.high_port = addr_pair.listen_addr.high_port;
  this->conn_addr.IP = addr_pair.conn_addr.IP;
  this->conn_addr.low_port = addr_pair.conn_addr.low_port;
  this->conn_addr.high_port = addr_pair.conn_addr.high_port;

  return;
}

mws_pkg_head::mws_pkg_head()
{
  this->filler_1[0] = 0xEE;
  this->filler_1[1] = 0xDD;
  this->filler_2[0] = 0xCC;
  this->filler_2[1] = 0xBB;
  this->filler_2[2] = 0xAA;

  return;
}

void mws_fd_detail::fd_init(bool option_del_deque)
{
  sess_addr_pair_t default_rcv_connection_setting;
  sockaddr_in_t default_sockaddr_in_t;

  bzero(&default_sockaddr_in_t, sizeof(default_sockaddr_in_t));
  default_sockaddr_in_t.sin_family = AF_INET;
  default_sockaddr_in_t.sin_port = htons(0);
  default_sockaddr_in_t.sin_addr.s_addr = inet_addr("0.0.0.0");

  this->fd = -1;
  this->role = FD_ROLE_UNKNOWN;
  this->status = FD_STATUS_UNKNOWN;
  this->src_listen_ptr = NULL;
  this->src_listen_addr_info = default_sockaddr_in_t;
  this->src_conn_ptr = NULL;
  this->src_conn_listen_addr_info = default_sockaddr_in_t;
  this->src_conn_rcv_addr_info = default_sockaddr_in_t;
  this->src_conn_sent_FC = false;
  this->src_conn_sent_topic_name = false;
  this->rcv_ptr = NULL;
  this->rcv_listen_addr_info = default_sockaddr_in_t;
  this->rcv_addr_info = default_sockaddr_in_t;
  this->rcv_sent_FD = false;
  this->rcv_sent_topic_name = false;
  this->rcv_connection_setting = default_rcv_connection_setting;

  if (option_del_deque == true)
  {
    if (this->msg_evq_ptr != NULL)
    {
      delete this->msg_evq_ptr;
      this->msg_evq_ptr = NULL;
    }
    if (this->send_buffer_ptr != NULL)
    {
      delete this->send_buffer_ptr;
      this->send_buffer_ptr = NULL;
    }
  }
  else
  {
    this->msg_evq_ptr = NULL;
    this->send_buffer_ptr = NULL;
  }

  this->msg_evq_number = 0;
  this->send_buffer_number = 0;

  this->last_heartbeat_time = time(NULL);

  return;
}

#if (MWS_DEBUG == 1)
  void mws_fd_detail::fd_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock fd:";
      log += std::to_string(this->fd);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&(this->mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking fd:";
      log += std::to_string(this->fd);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int mws_fd_detail::fd_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock fd:";
      log += std::to_string(this->fd);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&(this->mutex));
  }

  void mws_fd_detail::fd_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&(this->mutex));

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock fd:";
      log += std::to_string(this->fd);
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void mws_fd_detail::fd_lock()
  {
    pthread_mutex_lock(&(this->mutex));
    return;
  }

  int mws_fd_detail::fd_trylock()
  {
    return pthread_mutex_trylock(&(this->mutex));
  }

  void mws_fd_detail::fd_unlock()
  {
    pthread_mutex_unlock(&(this->mutex));
    return;
  }
#endif

time_t mws_fd_detail::get_last_heartbeat_time()
{
  return this->last_heartbeat_time;
}
