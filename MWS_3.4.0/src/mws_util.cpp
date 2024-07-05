//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_UTIL_CPP 1

#include <deque>
#include <map>
#include <string>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sstream>    // ostringstream.
#include <sys/time.h> // gettimeofday.
#include <netinet/tcp.h>
#include <algorithm>  // find().
#include <fcntl.h>
#include <pthread.h>

// DEBUG
#include <iostream>
#include <ctime>

#include "../inc/mws_init.h"
#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_socket.h"
#include "../inc/mws_time.h"
#include "../inc/mws_type_definition.h"
#include "../inc/mws_util.h"

namespace mws_util_global_variables
{
  static const int g_turn_on_tcp_nodelay = 1;
  static const int g_snd_recv_buffer_size = SND_RECV_BUFFER_SIZE;
  static const int g_turn_on_reuseaddr = 1;
}

using namespace mws_global_variable;
using namespace mws_log;
using namespace mws_util_global_variables;

int update_address_port(uint16_t& port, const ip_port_low_high_t addr);

// 功能: bind address.
//       IP 為 addr.IP,
//       port 為 addr.low_port 至 addr.high_port 其中之一.
// 回傳值 0: 正確完成.
// 回傳值 -1: bind 失敗.
// 參數 addr: 包含 conn 的 IP, low_port, high_port, 以及 next_bind_port.
// 參數 fd: 要 bind address 的 fd.
// 參數 &new_addr_info: bind 完成的 socket address info.
// 注意: 本函式沒有做 lock.
int bind_address(ip_port_low_high_t& addr,
                 fd_t fd,
                 sockaddr_in_t& new_addr_info)
{
  if ((addr.next_bind_port < addr.low_port) ||
      (addr.next_bind_port > addr.high_port))
  {
    addr.next_bind_port = addr.low_port;
  }

  // 在 addr.next_bind_port 到 addr.next_bind_port 間做一輪 bind 嘗試, 完成 bind 就可以 return.
  uint16_t start_port = addr.next_bind_port;
  bool flag_stop_trying = false;
  while (flag_stop_trying == false)
  {
    // Begin: set socket address info.
    sockaddr_in_t addr_info;
    {
      // socket address info.
      memset(&addr_info, 0x0, sizeof(sockaddr_in_t));
      addr_info.sin_family = AF_INET;
      addr_info.sin_addr.s_addr = inet_addr(addr.IP.c_str());
      addr_info.sin_port = htons(addr.next_bind_port);
    }
    // End: set socket address info.

    // Begin: bind and update port.
    {
      int rtv = mws_bind(fd,
                         (sockaddr_t*)&addr_info,
                         sizeof(addr_info));
      if (rtv == 0)
      {
        // 正確完成 bind, 更新 new_addr_info 並回傳 0.
        new_addr_info = addr_info;
        update_address_port(addr.next_bind_port, addr);
        return 0;
      }
      else
      {
        update_address_port(addr.next_bind_port, addr);
        // 所有可以嘗試的 port 都嘗試過了.
        if (start_port == addr.next_bind_port)
        {
          flag_stop_trying = true;
        }
      }
    }
    // End: bind and update port.
  }

  // 沒有完成 bind.
  return -1;
}

/*int socket_set_keepalive(int fd)
{
  // Set: use keepalive on fd.
  int alive = 1;
  if (setsockopt(fd, SOL_SOCKET, SO_KEEPALIVE, &alive, sizeof alive) != 0)
  {
    std::string log_body = "Set keepalive error: ";
    log_body += strerror(errno);
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    return -1;
  }

  // 5 秒鐘無數據, 觸發保活機制, 發送保活包.
  int idle = 5;
  if (setsockopt(fd, SOL_TCP, TCP_KEEPIDLE, &idle, sizeof idle) != 0)
  {
    std::string log_body = "Set keepalive idle error: ";
    log_body += strerror(errno);
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    return -1;
  }

  // 如果沒有收到回應, 則 3 秒鐘後重發保活包.
  int intv = 3;
  if (setsockopt(fd, SOL_TCP, TCP_KEEPINTVL, &intv, sizeof intv) != 0)
  {
    std::string log_body = "Set keepalive intv error: ";
    log_body += strerror(errno);
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    return -1;
  }

  // 連續 3 次沒收到保活包, 視為連接失效.
  int cnt = 3;
  if (setsockopt(fd, SOL_TCP, TCP_KEEPCNT, &cnt, sizeof cnt) != 0)
  {
    std::string log_body = "Set keepalive cnt error: ";
    log_body += strerror(errno);
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
    return -1;
  }

  return 0;
}*/

// 功能: 呼叫 socket() 取得 fd,
//       並做 TCP_NODELAY,
//       send buffer size,
//       receive buffer size 之設定.
// 回傳值 0: 正確完成.
// 回傳值 -1: 建立 socket 失敗 (呼叫方不用 close(fd)).
// 回傳值 -2: 設定 TCP_NODELAY 失敗.
// 回傳值 -3: 設定 send buffer size 失敗.
// 回傳值 -4: 設定 receive buffer size 失敗.
// 回傳值 -5: 設定 SO_REUSEADDR 失敗.
// 回傳值 -6: 設定 SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT 失敗.
// 參數 &fd: 要更新的 fd 變數.
// 注意: 本函式沒有做 lock.
int create_socket(fd_t& fd)
{
  // Begin: create socket.
  {
    fd = mws_socket(AF_INET, SOCK_STREAM, 0);
    if (fd == -1)
    {
      return -1;
    }
    else
    {
      if (g_mws_log_level >= 1)
      {
        std::string log_body_close = "create fd: " + std::to_string(fd);
        write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body_close);
      }
    }
  }
  // End: create connect socket.

  // Begin: 設定 TCP_NODELAY.
  {
    int rtv = setsockopt(fd,
                         IPPROTO_TCP,
                         TCP_NODELAY,
                         (char*)&g_turn_on_tcp_nodelay,
                         sizeof(int));
    if (rtv == -1)
    {
      return -2;
    }
  }
  // End: 設定 TCP_NODELAY.

  // Begin: 設定 SO_SNDBUF (send buffer size).
  {
    int rtv = setsockopt(fd,
                         SOL_SOCKET,
                         SO_SNDBUF,
                         &g_snd_recv_buffer_size,
                         sizeof(g_snd_recv_buffer_size));
    if (rtv == -1)
    {
      return -3;
    }
  }
  // End: 設定 設定 SO_SNDBUF (send buffer size).

  // Begin: 設定 SO_RCVBUF (receive buffer size).
  {
    int rtv = setsockopt(fd,
                         SOL_SOCKET,
                         SO_RCVBUF,
                         &g_snd_recv_buffer_size,
                         sizeof(g_snd_recv_buffer_size));
    if (rtv == -1)
    {
      return -4;
    }
  }
  // End: 設定 SO_RCVBUF (receive buffer size).

  // Begin: 設定 SO_REUSEADDR (allow reuse of local addresses).
  {
    int rtv = setsockopt(fd,
                         SOL_SOCKET,
                         SO_REUSEADDR,
                         (char*)&g_turn_on_reuseaddr,
                         sizeof(g_turn_on_reuseaddr));
    if (rtv == -1)
    {
      return -5;
    }
  }
  // End: 設定 SO_REUSEADDR (allow reuse of local addresses).

  // Begin: 設定 SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT.
  /*{
    if (socket_set_keepalive(fd) == (-1))
    {
      return -6;
    }
  }*/
  // End: 設定 SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT.

  return 0;
}

// 功能: 建立 src listen socket,
//       bind src address,
//       listen for incoming connection attempts.
// 回傳值 0: 正確完成.
// 回傳值 -1: 建立 socket 失敗.
// 回傳值 -2: 設定 TCP_NODELAY 失敗.
// 回傳值 -3: 設定 send buffer size 失敗.
// 回傳值 -4: 設定 receive buffer size 失敗.
// 回傳值 -5: bind port 都失敗.
// 回傳值 -6: 呼叫 listen() 失敗.
// 參數 *src_ptr: src 的指標.
int create_listen_socket(mws_src_t* src_ptr)
{
  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  std::string topic_name = src_ptr->topic_name;

  // Begin: create listen socket and set socket options.
  {
    int rtv = create_socket(src_ptr->src_listen_fd);
    if (rtv < 0)
    {
      std::string log_body;
      switch (rtv)
      {
        case -1:
        {
          // 無法建立 socket, 不用 close(fd).
          log_body = "mws_socket() failed ";
          break;
        }
        case -2:
        {
          // 無法完成設定 TCP_NODELAY, close(fd).
          mws_close(src_ptr->src_listen_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
          }

          log_body = "setsockopt(TCP_NODELAY) failed ";
          break;
        }
        case -3:
        {
          // 無法完成設定 send buffer size, close(fd).
          mws_close(src_ptr->src_listen_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
          }

          log_body = "setsockopt(SO_SNDBUF) failed ";
          break;
        }
        case -4:
        {
          // 無法完成設定 receive buffer size, close(fd).
          mws_close(src_ptr->src_listen_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
          }

          log_body = "setsockopt(SO_RCVBUF) failed ";
          break;
        }
        case -5:
        {
          // 無法完成設定 SO_REUSEADDR, close(fd).
          mws_close(src_ptr->src_listen_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
          }

          log_body = "setsockopt(SO_REUSEADDR) failed ";
          break;
        }
        /*case -6:
        {
          // 無法完成設定 SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT, close(fd).
          mws_close(src_ptr->src_listen_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
          }

          log_body = "setsockopt(SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT) failed ";
          break;
        }*/
        default:
        {
          log_body = "Unknown returned value ";
          break;
        }
      }
      log_body += "(rtv: ";
      log_body += std::to_string(rtv);
      log_body += ", errno: ";
      log_body += std::to_string(errno);
      log_body += ", strerr: ";
      log_body += strerror(errno);
      log_body += ")";
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return rtv;
    }
  }
  // End: create listen socket and set socket options.

  // Begin: bind listen address.
  {
    int rtv = bind_address(src_ptr->src_ip_port, src_ptr->src_listen_fd, src_ptr->src_listen_addr);
    // bind port 都失敗.
    if (rtv < 0)
    {
      std::string log_body =
        "mws_bind() failed for src(" + src_ptr->topic_name +
        ", " + src_ptr->src_ip_port.IP + ":" +
        std::to_string(src_ptr->src_ip_port.low_port) +
        "-" + std::to_string(src_ptr->src_ip_port.high_port) +
        ") (rtv: " + std::to_string(rtv) +
        ", errno: " + std::to_string(errno) +
        ", strerr: " + strerror(errno) + ")";
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
      // 無法完成 bind() 則 close socket.
      mws_close(src_ptr->src_listen_fd);
      if (g_mws_log_level >= 1)
      {
        std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
        write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
      }

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return -5;
    }
  }
  // End: bind listen address.

  // Begin: listen for incoming connection attempts.
  {
    // backlog = 3 表示最多可以有三個 pending connections.
    int rtv = mws_listen(src_ptr->src_listen_fd, 3);
    if (rtv < 0)
    {
      std::string log_body =
        "mws_listen() failed for src(" + src_ptr->topic_name +
        ", " + src_ptr->src_ip_port.IP + ":" +
        std::to_string(src_ptr->src_ip_port.low_port) +
        "-" + std::to_string(src_ptr->src_ip_port.high_port) +
        ") (rtv: " + std::to_string(rtv) +
        ", errno: " + std::to_string(errno) +
        ", strerr: " + strerror(errno) + ")";
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
      // 無法完成 listen() 則 close socket.
      mws_close(src_ptr->src_listen_fd);
      if (g_mws_log_level >= 1)
      {
        std::string log_body_close = "close src listen fd: " + std::to_string(src_ptr->src_listen_fd);
        write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
      }

      //pthread_mutex_unlock(&g_mws_global_mutex);
      #if (MWS_DEBUG == 1)
        g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_mws_global_mutex_unlock();
      #endif

      return -6;
    }
    else
    {
      std::string log_body =
        "mws_listen() success for src(" + src_ptr->topic_name +
        ", " + src_ptr->src_ip_port.IP + ":" +
        std::to_string(src_ptr->src_ip_port.low_port) + "-" +
        std::to_string(src_ptr->src_ip_port.high_port) + ") (" +
        std::to_string( ntohs(src_ptr->src_listen_addr.sin_port) ) + ") ";
      write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: listen for incoming connection attempts.

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  return 0;
}

// 功能: 檢查是否可以 connect 這個 listen address.
// 回傳值 0: 可以 connect 這個 listen address.
// 回傳值 -1: listen address 和 rcv 本身 bind 的 address 一樣.
// 回傳值 -2: listen address 和 rcv 設定中的已經連線的 address 一樣.
// 參數 rcv_listen_addr_info: 要檢查的 listen address.
// 參數 rcv_addr_info: rcv 本身 bind 的 address.
// 參數 rcv_list_connected_src_address: rcv 設定中的已經連線的 listen address.
int check_rcv_listen_sock_addr_info(sockaddr_in_t rcv_listen_addr_info,
                                    sockaddr_in_t rcv_addr_info,
                                    std::deque<sockaddr_in_t> rcv_list_connected_src_address)
{
  // Begin: 檢查 listen address 是否和 rcv 本身 bind 的 address 一樣.
  {
    if ((rcv_listen_addr_info.sin_addr.s_addr ==
         rcv_addr_info.sin_addr.s_addr) &&
        (rcv_listen_addr_info.sin_port ==
         rcv_addr_info.sin_port))
    {
      return -1;
    }
  }
  // End: 檢查 listen address 是否和 rcv 本身 bind 的 address 一樣.

  // Begin: 檢查 listen address 是否和 rcv 設定中的已經連線的 address 一樣.
  {
    for (size_t i = 0;
         i < rcv_list_connected_src_address.size();
         ++i)
    {
      // 和 rcv 的 cfg 的其中一個 session pair 設定中的已經連線的 address 相同.
      if ((rcv_listen_addr_info.sin_addr.s_addr ==
           rcv_list_connected_src_address[i].sin_addr.s_addr) &&
          (rcv_listen_addr_info.sin_port ==
           rcv_list_connected_src_address[i].sin_port))
      {
        return -2;
      }
    }
  }
  // End: 檢查 listen address 是否和 rcv 設定中的已經連線的 address 一樣.

  // 可以使用這個 address.
  return 0;
}

int create_connect_socket(wait_to_connect_rcv_session_t& sess_info,
                          sockaddr_in_t& rcv_listen_addr_info,
                          sockaddr_in_t& rcv_addr_info)
{
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " Start!!" << std::endl;
  //sleep(3);

  //pthread_mutex_lock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_lock();
  #endif

  mws_rcv_t* rcv_ptr = sess_info.rcv_ptr;
  std::string topic_name = rcv_ptr->topic_name;

  fd_t conn_fd = -1;

  // 第一次連線或是 connect 失敗需要重建 connect socket.
  bool flag_must_create_conn_socket = true;
  // 第幾輪 (0,1,2 ... ) 更新 rcv listen port (port_low to port_high).
  int update_listen_port_round = 0;

  while (update_listen_port_round < 2)
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " update_listen_port_round:"
    //          << std::to_string(update_listen_port_round) << std::endl;
    //sleep(1);

    // Begin: create connect socket and set socket options and bind rcv address.
    if (flag_must_create_conn_socket == true)
    {
      // Begin: create connect socket and set socket options.
      {
        int rtv = create_socket(conn_fd);
        if (rtv < 0)
        {
          std::string log_body;
          switch (rtv)
          {
            case -1:
            {
              // 無法建立 socket, 不用 close(fd).
              log_body = "mws_socket() failed ";
              break;
            }
            case -2:
            {
              // 無法完成設定 TCP_NODELAY, close(fd).
              mws_close(conn_fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body_close = "close rcv fd: " + std::to_string(conn_fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
              }

              log_body = "setsockopt(TCP_NODELAY) failed ";
              break;
            }
            case -3:
            {
              // 無法完成設定 send buffer size, close(fd).
              mws_close(conn_fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body_close = "close rcv fd: " + std::to_string(conn_fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
              }

              log_body = "setsockopt(SO_SNDBUF) failed ";
              break;
            }
            case -4:
            {
              // 無法完成設定 receive buffer size, close(fd).
              mws_close(conn_fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body_close = "close rcv fd: " + std::to_string(conn_fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
              }

              log_body = "setsockopt(SO_RCVBUF) failed ";
              break;
            }
            case -5:
            {
              // 無法完成設定 SO_REUSEADDR, close(fd).
              mws_close(conn_fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body_close = "close rcv fd: " + std::to_string(conn_fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
              }

              log_body = "setsockopt(SO_REUSEADDR) failed ";
              break;
            }
            /*case -6:
            {
              // 無法完成設定 SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT, close(fd).
              mws_close(conn_fd);
              if (g_mws_log_level >= 1)
              {
                std::string log_body_close = "close rcv fd: " + std::to_string(conn_fd);
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
              }

              log_body = "setsockopt(SO_KEEPALIVE, TCP_KEEPIDLE, TCP_KEEPINTVL, TCP_KEEPCNT) failed ";
              break;
            }*/
            default:
            {
              log_body = "Unknown returned value ";
              break;
            }
          }
          log_body += "(rtv: ";
          log_body += std::to_string(rtv);
          log_body += ", errno: ";
          log_body += std::to_string(errno);
          log_body += ", strerr: ";
          log_body += strerror(errno);
          log_body += ")";
          write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

          memset(&rcv_listen_addr_info, 0, sizeof(rcv_listen_addr_info));
          memset(&rcv_addr_info, 0, sizeof(rcv_addr_info));

          //pthread_mutex_unlock(&g_mws_global_mutex);
          #if (MWS_DEBUG == 1)
            g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_mws_global_mutex_unlock();
          #endif

          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !! return " << std::endl;
          //sleep(1);

          return rtv;
        }
      }
      // End: create connect socket and set socket options.

      // Begin: bind rcv address.
      {
        int rtv = bind_address(sess_info.rcv_connection_setting.conn_addr, conn_fd, rcv_addr_info);
        // bind port 都失敗.
        if (rtv < 0)
        {
          std::string log_body =
            "mws_bind() failed for src(" + rcv_ptr->topic_name +
            ", " + sess_info.rcv_connection_setting.conn_addr.IP + ":" +
            std::to_string(sess_info.rcv_connection_setting.conn_addr.low_port) +
            "-" + std::to_string(sess_info.rcv_connection_setting.conn_addr.high_port) +
            ") (rtv: " + std::to_string(rtv) +
            ", errno: " + std::to_string(errno) +
            ", strerr: " + strerror(errno) + ")";
          write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);
          // 無法完成 bind() 則 close socket.
          mws_close(conn_fd);
          if (g_mws_log_level >= 1)
          {
            std::string log_body = "close rcv fd: " + std::to_string(conn_fd);
            write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
          }

          memset(&rcv_listen_addr_info, 0, sizeof(rcv_listen_addr_info));
          memset(&rcv_addr_info, 0, sizeof(rcv_addr_info));

          //pthread_mutex_unlock(&g_mws_global_mutex);
          #if (MWS_DEBUG == 1)
            g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_mws_global_mutex_unlock();
          #endif

          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !! bind fail , return "<< std::endl;
          //sleep(1);

          return -5;
        }
      }
      // End: bind rcv address.

      //std::string log_body =
      //  "rcv bind addr. (" + sess_info.rcv_connection_setting.conn_addr.IP + ":" + std::to_string( ntohs(rcv_addr_info.sin_port) ) + ")";
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << log_body << std::endl;
      //sleep(1);

      flag_must_create_conn_socket = false;
    }
    // End: create connect socket and set socket options and bind rcv address.

    // Begin: initialize rcv listen socket address info.
    {
      memset(&rcv_listen_addr_info, 0x0, sizeof(sockaddr_in_t));
      rcv_listen_addr_info.sin_family = AF_INET;
      rcv_listen_addr_info.sin_addr.s_addr = inet_addr(sess_info.rcv_connection_setting.listen_addr.IP.c_str());
      rcv_listen_addr_info.sin_port = htons(sess_info.next_port);
    }
    // End: initialize rcv listen socket address info.

    // Begin: check rcv listen socket address info and connect to src.
    {
      int flag_rcv_listen_sock_addr_ok =
        check_rcv_listen_sock_addr_info(rcv_listen_addr_info,
                                        rcv_addr_info,
                                        rcv_ptr->rcv_list_connected_src_address);
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
      //          << " check_rcv_listen_sock_addr_info:" << std::to_string(flag_rcv_listen_sock_addr_ok)
      //          << std::endl;
      if (flag_rcv_listen_sock_addr_ok == 0)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " rcv_listen_sock_addr_ok" << std::endl;
        // Begin: connect to src.
        {
          int rtv = mws_connect(conn_fd,
                                (sockaddr_t*)&rcv_listen_addr_info,
                                sizeof(rcv_listen_addr_info));
          if (rtv == 0)
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " mws_connect() rtv == 0" << std::endl;
            //#ifdef __TANDEM
            {
              // *** check socket status.
              // when using connect() in OSS, connect()
              // may fail with rtv == 0 and errno == 0,
              // but error_code (checks socket's status) != 0.
              int error_code = 0;
              socklen_t error_code_size = (socklen_t)sizeof(error_code);
              getsockopt(conn_fd,
                         SOL_SOCKET,
                         SO_ERROR,
                         &error_code,
                         &error_code_size);
              if (error_code != 0)
              {
                //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " mws_connect() error_code != 0" << std::endl;

                // connect 回傳成功但實際上 socket 已經發生錯誤, 需要重建 socket.
                rtv = -2;
              }
            }
            //#endif // __TANDEM
          }
          else
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
          }

          // connect() 發生錯誤.
          if (rtv != 0)
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
            //          << " mws_connect() rtv != 0 "
            //          << " errno: " << std::to_string( errno )
            //          << " " << strerror( errno )
            //          << std::endl;

            // connect 失敗, connect socket 需要重建.
            flag_must_create_conn_socket = true;

            // connect 失敗, 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
            rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(conn_fd);
            if (g_mws_log_level >= 1)
            {
              std::string log_body = "push to ctx_list_wait_to_close_rcv_fds-rcv fd: " + std::to_string(conn_fd);
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " call ctx_list_wait_to_close_rcv_fds.push_back() fd:" << std::to_string(conn_fd) << std::endl;

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
            //          << " sess_info.next_port:" << std::to_string(sess_info.next_port)
            //          << std::endl;
            int rtv = update_address_port(sess_info.next_port, sess_info.rcv_connection_setting.listen_addr);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
            //          << " sess_info.next_port:" << std::to_string(sess_info.next_port)
            //          << " rtv:" << std::to_string(rtv)
            //          << std::endl;
            if (rtv == 1)
            {
              ++update_listen_port_round;
            }
          }
          else
          {
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;

            // 正確完成 connect() 動作.
            // Begin: 更新 table.
            {
              // mws_rcv::rcv_connect_fds
              rcv_ptr->rcv_connect_fds.push_back(conn_fd);
              // mws_rcv::rcv_list_connected_src_address
              rcv_ptr->rcv_list_connected_src_address.push_back(rcv_listen_addr_info);
            }

            //if (g_mws_log_level >= 1)
            {
              std::string log_body = "get rcv fd: " + std::to_string(conn_fd);
              log_body += "(";
              log_body += sess_info.rcv_connection_setting.listen_addr.IP.c_str();
              log_body += ":";
              log_body += std::to_string(sess_info.next_port);
              log_body += ")";
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }

            // End: 更新 table.
            //pthread_mutex_unlock(&g_mws_global_mutex);
            #if (MWS_DEBUG == 1)
              g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
            #else
              g_mws_global_mutex_unlock();
            #endif

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
            //sleep(1);

            return conn_fd;
          }
        }
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " rcv_listen_sock_addr_ok" << std::endl;
        // End: connect to src.
      }
      else
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
        //          << " sess_info.next_port:" << std::to_string(sess_info.next_port)
        //          << std::endl;
        int rtv = update_address_port(sess_info.next_port, sess_info.rcv_connection_setting.listen_addr);
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
        //          << " sess_info.next_port:" << std::to_string(sess_info.next_port)
        //          << std::endl;
        if (rtv == 1)
        {
          //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;

          ++update_listen_port_round;
        }
      }

      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
    }
    // End: check rcv listen socket address info and connect to src.

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " " << std::endl;
  } // while (update_listen_port_round < 2)

  memset(&rcv_listen_addr_info, 0, sizeof(rcv_listen_addr_info));
  memset(&rcv_addr_info, 0, sizeof(rcv_addr_info));

  //pthread_mutex_unlock(&g_mws_global_mutex);
  #if (MWS_DEBUG == 1)
    g_mws_global_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_mws_global_mutex_unlock();
  #endif

  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << std::endl;
  //sleep(1);

  // rcv listen port low to high 都無法完成 connect.
  return -6;
}

void step_accept_connection(mws_ctx_t* ctx_ptr, fd_t src_listen_fd)
{
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " begin accept fd:" << std::to_string(src_listen_fd) << std::endl;

  #if (MWS_DEBUG == 1)
    g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_lock();
  #endif

  #if (MWS_DEBUG == 1)
    g_fd_table[src_listen_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_fd_table[src_listen_fd].fd_lock();
  #endif

  std::string topic_name = g_fd_table[src_listen_fd].src_listen_ptr->topic_name;

  // Begin: g_mws_log_level >= 1 時, 刷進入函式的 log.
  if (g_mws_log_level >= 1)
  {
    std::string log_body_role_addr_info;
    fd_info_log(src_listen_fd, log_body_role_addr_info);
    std::string log_body = "step_accept_connection() start. " + log_body_role_addr_info;
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }
  // End: g_mws_log_level >= 1 時, 刷進入函式的 log.

  // Begin: 執行 accept connection.
  {
    // rcv 的 socket address info.
    sockaddr_in_t rcv_addr_info;
    socklen_t rcv_addr_info_len = (socklen_t)sizeof(sockaddr_in_t);
    memset(&rcv_addr_info, 0x0, sizeof(rcv_addr_info));

    fd_t src_conn_fd = mws_accept(src_listen_fd, (sockaddr_t*)&rcv_addr_info, &rcv_addr_info_len);

    // Begin: accept() 失敗.
    if (src_conn_fd < 0)
    {
      std::string log_body_role_addr_info;
      fd_info_log(src_listen_fd, log_body_role_addr_info);
      std::string log_body =
        "mws_accept() failed for src(" +
        log_body_role_addr_info +
        ") (rtv: " + std::to_string(src_conn_fd) +
        ", errno: " + std::to_string(errno) +
        ", strerr: " + strerror(errno) + ")";
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

      #if (MWS_DEBUG == 1)
        g_fd_table[src_listen_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[src_listen_fd].fd_unlock();
      #endif

      #if (MWS_DEBUG == 1)
        g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_unlock();
      #endif

      return;
    }
    else
    {
      if (g_mws_log_level >= 1)
      {
        std::string log_body_close = "create fd: " + std::to_string(src_conn_fd);
        write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body_close);
      }
    }
    // End: accept() 失敗.

    // Begin: 正確完成 accept().
    {
      // Begin: 維護各物件變數.
      {
        #if (MWS_DEBUG == 1)
          g_fd_table[src_conn_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[src_conn_fd].fd_lock();
        #endif

        //src: std::vector<fd_t> src_connect_fds;
        g_fd_table[src_listen_fd].src_listen_ptr->src_connect_fds.push_back(src_conn_fd);

        // ctx: std::deque<wait_to_check_topic_src_conn_session_t> ctx_list_wait_to_check_topic_src_conn_session;
        wait_to_check_topic_src_conn_session_t temp_obj;
        temp_obj.fd = src_conn_fd;
        temp_obj.src_ptr = g_fd_table[src_listen_fd].src_listen_ptr;
        temp_obj.src_conn_listen_addr_info = g_fd_table[src_listen_fd].src_listen_addr_info;
        temp_obj.src_conn_rcv_addr_info = rcv_addr_info;
        g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.push_back(temp_obj);

        // ctx: std::deque<fd_t> ctx_list_owned_src_conn_fds;
        //pthread_mutex_lock(&(g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
        #if (MWS_DEBUG == 1)
          g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex_lock();
        #endif

        g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds.push_back(src_conn_fd);

        //pthread_mutex_unlock(&(g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
        #if (MWS_DEBUG == 1)
          g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[src_listen_fd].src_listen_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex_unlock();
        #endif

        // 新增 src_conn_fd 至 all_set.
        FD_SET(src_conn_fd, &ctx_ptr->all_set);
        {
          std::string log_body = "Add src conn fd:" +
                                 std::to_string(src_conn_fd) +
                                 " into all_set ";
          write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
        }

        // 更新 ctx::max_fd
        ctx_ptr->update_max_fd(src_conn_fd);

        // Begin: maintain g_fd_table
        {
          g_fd_table[src_conn_fd].fd = src_conn_fd;
          g_fd_table[src_conn_fd].role = FD_ROLE_SRC_CONN;
          update_g_fd_table_status(src_conn_fd,
                                   FD_STATUS_SRC_CONN_PREPARE,
                                   __func__,
                                   __LINE__);
          g_fd_table[src_conn_fd].src_conn_ptr = g_fd_table[src_listen_fd].src_listen_ptr;
          g_fd_table[src_conn_fd].src_conn_listen_addr_info = g_fd_table[src_listen_fd].src_listen_addr_info;
          g_fd_table[src_conn_fd].src_conn_rcv_addr_info = rcv_addr_info;
          g_fd_table[src_conn_fd].msg_evq_ptr = new mws_fast_deque_t(sizeof(mws_msg_evq_buffer_t), SRC_CONN_NUM_OF_DEFAULT_RECV_BUFFER_BLOCK, SRC_CONN_NUM_OF_EXT_RECV_BUFFER_BLOCK);
          g_fd_table[src_conn_fd].msg_evq_number = g_fd_table[src_conn_fd].msg_evq_ptr->get_new_deque();
          g_fd_table[src_conn_fd].send_buffer_ptr = new mws_fast_deque_t(sizeof(mws_send_buff_t), SRC_CONN_NUM_OF_DEFAULT_SND_BUFFER_BLOCK, SRC_CONN_NUM_OF_EXT_SND_BUFFER_BLOCK);
          g_fd_table[src_conn_fd].send_buffer_number = g_fd_table[src_conn_fd].send_buffer_ptr->get_new_deque();
        }
        // End: maintain g_fd_table

        // Begin: 更新 evq_list_owned_fds.
        {
          g_fd_table[src_conn_fd].src_conn_ptr->evq_ptr->evq_list_owned_fds.push_back(src_conn_fd);
        }
        // End: 更新 evq_list_owned_fds.

        #if (MWS_DEBUG == 1)
          g_fd_table[src_conn_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[src_conn_fd].fd_unlock();
        #endif
      }
      // End: 維護各物件變數.

      // Begin: 刷 accept 正確完成 log.
      {
        std::string log_body_role_addr_info;
        fd_info_log(src_conn_fd, log_body_role_addr_info);
        std::string log_body = "mws_accept() success for src(" + log_body_role_addr_info + ")";
        write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
      }
      // End: 刷 accept 正確完成 log.

      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " accept ok src_conn_fd:" << std::to_string(src_conn_fd) << std::endl;
    }
    // End: 正確完成 accept().
  }
  // End: 執行 accept connection.

  #if (MWS_DEBUG == 1)
    g_fd_table[src_listen_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_fd_table[src_listen_fd].fd_unlock();
  #endif

  #if (MWS_DEBUG == 1)
    g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
  #else
    g_fd_table[src_listen_fd].src_listen_ptr->evq_ptr->evq_unlock();
  #endif

  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " end accept fd:" << std::to_string(src_listen_fd) << std::endl;

  return;
}

void set_port_high_low(ip_port_low_high_t& dest,
                       uint16_t port_low,
                       uint16_t port_high,
                       const std::string source_file_of_caller,
                       const std::string function_of_caller,
                       const int line_no_of_caller)
{
  if (port_high < port_low)
  {
    uint16_t temp = port_high;
    port_high = port_low;
    port_low = temp;
    std::string log_body =
                  "Port low and high are reversed.(" +
                  std::to_string(port_low) +
                  "/" +
                  std::to_string(port_high) +
                  ")" +
                  "(" +
                  source_file_of_caller +
                  "/" +
                  function_of_caller +
                  "/" +
                  std::to_string(line_no_of_caller) +
                  ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  if (port_low < 1)
  {
    uint16_t temp = port_low;
    port_low = 1;
    std::string log_body =
                  "Port low < 1, reset to 1.(" +
                  std::to_string(temp) +
                  ")" +
                  "(" +
                  source_file_of_caller +
                  "/" +
                  function_of_caller +
                  "/" +
                  std::to_string(line_no_of_caller) +
                  ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  if (port_high > 65535)
  {
    uint16_t temp = port_high;
    port_high = 65535;
    std::string log_body =
                  "Port high > 65535, reset to 65535.(" +
                  std::to_string(temp) +
                  ")" +
                  "(" +
                  source_file_of_caller +
                  "/" +
                  function_of_caller +
                  "/" +
                  std::to_string(line_no_of_caller) +
                  ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  dest.low_port = port_low;
  dest.high_port = port_high;
  dest.next_bind_port = port_low;

  return;
}

// 功能: 更新要使用的 address 的 port.
// 回傳值 0: 更新 port 完成.
// 回傳值 1: 更新 port 完成且已經從 low_port 試到 high_port.
// 參數 &port: 要更新的 port.
// 參數 addr: 用來取得 low_port 和 high_port 的 addr.
int update_address_port(uint16_t& port, const ip_port_low_high_t addr)
{
  // 當前 port == high_port, 將 port 轉為 low_port.
  if (port >= addr.high_port)
  {
    port = addr.low_port;
    // 當作已經從 low_port 試到 high_port.
    return 1;
  }
  else
  {
    ++port;
    return 0;
  }
}

// 功能: 將 sockaddr_in_t 內的 IP/port 分別轉為 std::string.
// 回傳值: 無.
// 參數 addr_info: sockaddr_in_t 格式的來源變數.
// 參數 &str_ip: std::string 格式的 IP.
// 參數 &str_port: std::string 格式的 port.
void sockaddr_in_t_to_string(const sockaddr_in_t addr_info,
                             std::string& str_ip,
                             std::string& str_port)
{
  str_ip = inet_ntoa(addr_info.sin_addr);
  str_port = std::to_string(ntohs(addr_info.sin_port));

  return;
}

// 功能: 根據 fd 的 role 產生要寫的 log.
// 回傳值: 無.
// 參數 fd: 要寫 log 的 fd 值.
// 參數 &log_body: std::string 格式的 log.
void fd_info_log(const fd_t fd,
                 std::string& log_body)
{
  str_ip_port_t src_listen_addr;
  str_ip_port_t rcv_addr;

  if (g_fd_table[fd].role == FD_ROLE_SRC_CONN)
  {
    sockaddr_in_t_to_string(g_fd_table[fd].src_conn_listen_addr_info,
                            src_listen_addr.str_ip,
                            src_listen_addr.str_port);
    sockaddr_in_t_to_string(g_fd_table[fd].src_conn_rcv_addr_info,
                            rcv_addr.str_ip,
                            rcv_addr.str_port);

    log_body =
      "FD_ROLE_SRC_CONN (fd:" + std::to_string(fd) +
      //"), topic name (" +
      //g_fd_table[fd].src_conn_ptr->topic_name +
      "), src listen addr. (" + src_listen_addr.str_ip +
      ":" + src_listen_addr.str_port +
      "), rcv addr. (" + rcv_addr.str_ip +
      ":" + rcv_addr.str_port + ")";
  }
  else if (g_fd_table[fd].role == FD_ROLE_RCV)
  {
    sockaddr_in_t_to_string(g_fd_table[fd].rcv_listen_addr_info,
                            src_listen_addr.str_ip,
                            src_listen_addr.str_port);
    sockaddr_in_t_to_string(g_fd_table[fd].rcv_addr_info,
                            rcv_addr.str_ip,
                            rcv_addr.str_port);
    log_body =
      "FD_ROLE_RCV (fd:" + std::to_string(fd) +
      //"), topic name (" +
      //g_fd_table[fd].rcv_ptr->topic_name +
      "), src listen addr. (" + src_listen_addr.str_ip +
      ":" + src_listen_addr.str_port +
      "), rcv addr. (" + rcv_addr.str_ip +
      ":" + rcv_addr.str_port + ")";
  }
  else if (g_fd_table[fd].role == FD_ROLE_SRC_LISTEN)
  {
    sockaddr_in_t_to_string(g_fd_table[fd].src_listen_addr_info,
                            src_listen_addr.str_ip,
                            src_listen_addr.str_port);
    log_body =
      "FD_ROLE_SRC_LISTEN (fd:" + std::to_string(fd) +
      //"), topic name (" +
      //g_fd_table[fd].src_listen_ptr->topic_name +
      "), src listen addr. (" + src_listen_addr.str_ip +
      ":" + src_listen_addr.str_port + ")";
  }
  else
  {
    // error...
  }

  return;
}

ssize_t recv_data(void* recv_buff_ptr,
                  const fd_t fd,
                  const size_t max_len,
                  int32_t max_retry_cnt)
{
  int flags = 0;

  #ifdef __TANDEM
  {
    // Begin: turn on O_NONBLOCK.
    {
      // set fd_flag to O_NONBLOCK since OSS cannot use MSG_DONTWAIT.
      // hex: 0002 (O_RDWR).
      // hex: 0800 (O_NONBLOCK).
      int fd_flag = fcntl(fd, F_GETFL, 0);
      fcntl(fd, F_SETFL, fd_flag | O_NONBLOCK);
    }
    // End: turn on O_NONBLOCK.
  }
  #else
  {
    flags |= MSG_DONTWAIT;
  }
  #endif

  ssize_t rtv = 0;
  int32_t try_cnt = 0;
  bool flag_try_to_recv = true;
  do
  {
    // receive data from socket.
    rtv = mws_recv(fd, recv_buff_ptr, max_len, flags);
    if (rtv > 0)
    {
      flag_try_to_recv = false;
    }
    else if (rtv == 0)
    {
      if (g_mws_log_level >= 1)
      {
        std::string log_body = "mws_recv() fail (fd:" + std::to_string(fd) + ", rtv == 0, the peer is closed)";
        write_to_log("", 99, "D", __FILE__, __func__, __LINE__, log_body);
      }

      flag_try_to_recv = false;
    }
    //else if (rtv == (-1))
    else
    {
      if ((errno == EAGAIN) || (errno == EWOULDBLOCK) || (errno == EINTR))
      {
        ++try_cnt;
        if (try_cnt > max_retry_cnt)
        {
          rtv = (-2);
          flag_try_to_recv = false;
        }
        else
        {
          usleep(50);
          //flag_try_to_recv = true;
        }
      }
      else
      {
        std::string log_body =
          "mws_recv() fail (fd:" + std::to_string(fd) +
          "), errno: " + std::to_string(errno) +
          "(" + strerror(errno) + ")";
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        flag_try_to_recv = false;
      }
    }
  }
  while (flag_try_to_recv == true);

  #ifdef __TANDEM
  {
    // Begin: turn off O_NONBLOCK.
    {
      int fd_flag = fcntl(fd, F_GETFL, 0);
      fcntl(fd, F_SETFL, fd_flag & ~O_NONBLOCK);
    }
    // End: turn off O_NONBLOCK.
  }
  #endif

  return rtv;
}

// topic check code 完全傳送完畢, 回傳 rtv (等於 max_len).
// topic check code 沒有完全傳送完畢或傳送失敗, 回傳 -1.
ssize_t send_topic_check_code(void* send_buff_ptr,
                              const fd_t fd,
                              const size_t max_len)
{
  char* curr_ptr = (char*)send_buff_ptr;

  ssize_t rtv = mws_send_nonblock(fd,
                                  curr_ptr,
                                  max_len,
                                  0);

  if (rtv == (ssize_t)max_len)
  {
    return rtv;
  }

  return -1;
}

// topic name 完全傳送完畢, 回傳 rtv (等於 g_size_of_mws_topic_name_msg_t).
// topic name 沒有完全傳送完畢或傳送失敗, 回傳 -1.
ssize_t send_topic_name(const mws_topic_name_msg_t topic_name_msg,
                        const fd_t fd)
{
  char* curr_ptr = (char*)&topic_name_msg;

  ssize_t rtv = mws_send_nonblock(fd,
                                  curr_ptr,
                                  g_size_of_mws_topic_name_msg_t,
                                  0);

  if (rtv == (ssize_t)g_size_of_mws_topic_name_msg_t)
  {
    return rtv;
  }

  return -1;
}

void rcv_topic_check_error(std::deque<fd_t>::iterator& it,
                           const std::string function,
                           const int line_no)
{
  std::string topic_name = g_fd_table[*it].rcv_ptr->topic_name;

  // Begin: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].rcv_ptr->erase_rcv_list_connected_src_address(g_fd_table[*it].rcv_listen_addr_info);
    if (rtv != 0)
    {
      str_ip_port_t rcv_listen_addr;
      sockaddr_in_t_to_string(g_fd_table[*it].rcv_listen_addr_info,
                              rcv_listen_addr.str_ip,
                              rcv_listen_addr.str_port);

      // rcv_list_connected_src_address 沒有該 sockaddr_in_t 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) +
                 "(" + rcv_listen_addr.str_ip + ":" + rcv_listen_addr.str_port +
                 ") does not exist in rcv_listen_addr_info";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.

  // Begin: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].rcv_ptr->erase_rcv_connect_fds(*it);
    if (rtv != 0)
    {
      // rcv_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in rcv_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.

  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  {
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[*it].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " evq_list_owned_fds.size():" << std::to_string(g_fd_table[*it].rcv_ptr->evq_ptr->evq_list_owned_fds.size()) << std::endl;
    int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->erase_evq_list_owned_fds(*it);
    if (rtv != 0)
    {
      //std::cout << std::string(__func__) << std::endl;
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[*it].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[*it].rcv_ptr->evq_ptr->evq_unlock();
    #endif
  }
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].rcv_ptr->ctx_ptr->erase_ctx_list_wait_to_check_topic_rcv_session(*it);
    if (rtv != 0)
    {
      // ctx_list_wait_to_check_topic_rcv_session 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in ctx_list_wait_to_check_topic_rcv_session";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_owned_rcv_fds, 所以要用 it 直接刪除.
    //pthread_mutex_lock(&(g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds_mutex));
    g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds.erase(it);
    //pthread_mutex_unlock(&(g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds_mutex));
  }
  // End: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(*it, &g_fd_table[*it].rcv_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove rcv fd:" +
                           std::to_string(*it) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_WAIT_TO_CLOSE.
  //g_fd_table[*it].status = FD_STATUS_RCV_WAIT_TO_CLOSE;
  update_g_fd_table_status(*it,
                           FD_STATUS_RCV_WAIT_TO_CLOSE,
                           __func__,
                           __LINE__);

  // Begin: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  {
    // 取得下次連線用的 port.
    uint16_t next_port = ntohs(g_fd_table[*it].rcv_listen_addr_info.sin_port);
    update_address_port(next_port, g_fd_table[*it].rcv_connection_setting.listen_addr);

    wait_to_connect_rcv_session_t temp_obj;
    temp_obj.rcv_ptr = g_fd_table[*it].rcv_ptr;
    temp_obj.rcv_connection_setting = g_fd_table[*it].rcv_connection_setting;
    temp_obj.next_port = next_port;
    temp_obj.try_cnt = 0;
    g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
  }
  // End: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.

  // 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
  g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(*it);
  if (g_mws_log_level >= 1)
  {
    std::string log_body = "push to ctx_list_wait_to_close_rcv_fds-rcv fd: " + std::to_string(*it);
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }

  std::string log_body_role_addr_info;
  fd_info_log(*it, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  //std::cout << __func__ << ":" << __LINE__ << " " << log_body << std::endl;

  return;
}

void step_rcv_wait_fefc(std::deque<fd_t>::iterator& it)
{
  bool flag_break_while_loop = false;
  while (flag_break_while_loop == false)
  {
    char rcv_buff[1];
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf call recv_data" << std::endl;
    ssize_t rtv = recv_data((void*)&rcv_buff[0], *it, 1, MAX_NUM_OF_RETRIES_RECV_FFFD_FEFC);
    if (rtv > 0)
    {
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " recv_data rtv > 0" << std::endl;
      if (rcv_buff[0] == (char)0xFE)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " (rcv_buff[0] == (char)0xFE)" << std::endl;
        // 將 g_fd_table 的 status 改為 FD_STATUS_RCV_WAIT_FC.
        //g_fd_table[*it].status = FD_STATUS_RCV_WAIT_FC;
        update_g_fd_table_status(*it,
                                 FD_STATUS_RCV_WAIT_FC,
                                 __func__,
                                 __LINE__);

        // Begin: send FD to src.
        {
          // 只能送一次 0xFD.
          if (g_fd_table[*it].rcv_sent_FD == false)
          {
            // 傳送 0xFD.
            char send_buff[1];
            send_buff[0] = (char)0xFD;
            ssize_t rtv = send_topic_check_code((void*)&send_buff[0], *it, 1);
            if (rtv < 0)
            {
              //std::cout << __func__ << ":" << __LINE__ << " !!! Sent FD to src error" << std::endl;
              //sleep(5);

              // fd 發生問題, 要斷線.
              rcv_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].rcv_sent_FD = true;

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !!! Sent FD to src" << std::endl;
              //sleep(3);
            }
          }
        }
        // End: send FD to src.
      }
      else if (rcv_buff[0] == (char)0xFC)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " (rcv_buff[0] == (char)0xFC)" << std::endl;

        flag_break_while_loop = true;

        // 將 g_fd_table 的 status 改為 FD_STATUS_RCV_WAIT_TOPIC_NAME.
        //g_fd_table[*it].status = FD_STATUS_RCV_WAIT_TOPIC_NAME;
        update_g_fd_table_status(*it,
                                 FD_STATUS_RCV_WAIT_TOPIC_NAME,
                                 __func__,
                                 __LINE__);

        // Begin: send FD to src.
        {
          // 只能送一次 0xFD.
          if (g_fd_table[*it].rcv_sent_FD == false)
          {
            // 傳送 0xFD.
            char send_buff[1];
            send_buff[0] = (char)0xFD;
            ssize_t rtv = send_topic_check_code((void*)&send_buff[0], *it, 1);
            if (rtv < 0)
            {
              //std::cout << __func__ << ":" << __LINE__ << " !!! Sent FD to src error" << std::endl;
              //sleep(5);

              // fd 發生問題, 要斷線.
              rcv_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].rcv_sent_FD = true;

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !!! Sent FD to src" << std::endl;
              //sleep(3);
            }
          }
        }
        // End: send FD to src.

        // Begin: send topic name to src.
        {
          // 只能送一次 topic name.
          if (g_fd_table[*it].rcv_sent_topic_name == false)
          {
            // 傳送 topic name.
            mws_topic_name_msg_t temp_obj;
            memset(&temp_obj, 0, g_size_of_mws_topic_name_msg_t);
            temp_obj.topic_name_len = htons((uint16_t)g_fd_table[*it].rcv_ptr->topic_name.size());
            memcpy((void*)(&temp_obj.topic_name[0]),
                   (void*)(g_fd_table[*it].rcv_ptr->topic_name.c_str()),
                   g_fd_table[*it].rcv_ptr->topic_name.size());
            ssize_t rtv = send_topic_name(temp_obj, *it);
            if (rtv < 0)
            {
              //std::cout << __func__ << ":" << __LINE__ << " !!! Sent topic name to src error" << std::endl;
              //sleep(5);

              // fd 發生問題, 要斷線.
              rcv_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].rcv_sent_topic_name = true;

              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !!! Sent topic name to src ok" << std::endl;
              //sleep(3);
            }
          }
        }
        // End: send topic name to src.
      }
    }
    else if ((rtv == 0) || (rtv == -1))
    {
      //std::cout << __func__ << ":" << __LINE__ << " recv_data error" << std::endl;
      //sleep(5);

      flag_break_while_loop = true;
      // fd 發生問題, 要斷線.
      rcv_topic_check_error(it, __func__, __LINE__);
    }
    // EAGAIN or EWOULDBLOCK or EINTR.
    else if (rtv == (-2))
    {
      flag_break_while_loop = true;
    }
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af call recv_data" << std::endl;
  } // while (flag_break_while_loop == false)

  return;
}

void step_rcv_wait_topic_name(std::deque<fd_t>::iterator& it)
{
  std::string topic_name = g_fd_table[*it].rcv_ptr->topic_name;

  // 累計接收到的 byte 數.
  size_t accu_rcv_size = 0;
  // 接收資料用的 buffer.
  //char rcv_buff[g_size_of_mws_topic_name_msg_t]; // 無法這樣宣告.
  char rcv_buff[MAX_MSG_SIZE];

  bool flag_break_while_loop = false;
  while (flag_break_while_loop == false)
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf call recv_data" << std::endl;
    ssize_t rtv = recv_data((void*)&rcv_buff[accu_rcv_size],
                            *it,
                            (g_size_of_mws_topic_name_msg_t - accu_rcv_size),
                            MAX_NUM_OF_RETRIES_RECV_TOPIC_NAME);
    if (rtv > 0)
    {
      //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " step_rcv_wait_topic_name read data ok" << std::endl;

      accu_rcv_size += rtv;
      if (accu_rcv_size == g_size_of_mws_topic_name_msg_t)
      {
        //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " accu_rcv_size == g_size_of_mws_topic_name_msg_t" << std::endl;

        flag_break_while_loop = true;
        // Begin: 檢核 topic name.
        {
          mws_topic_name_msg_t temp_obj;
          memcpy((void*)&temp_obj, &rcv_buff[0], g_size_of_mws_topic_name_msg_t);
          std::string src_topic_name(&temp_obj.topic_name[0], ntohs(temp_obj.topic_name_len));
          //std::cout << "temp_obj.topic_name_len:" << std::to_string( ntohs(temp_obj.topic_name_len) ) << std::endl;
          //std::cout << "src_topic_name:" << src_topic_name << std::endl;
          //std::cout << "g_fd_table[*it].rcv_ptr->topic_name:" << g_fd_table[*it].rcv_ptr->topic_name << std::endl;
          if (src_topic_name == g_fd_table[*it].rcv_ptr->topic_name)
          {
            // topic name 檢核正確.
            // 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.
            // 產生 MWS_MSG_BOS.
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf create MWS_MSG_BOS" << std::endl;
            mws_event_t* event_ptr = g_fd_table[*it].rcv_ptr->evq_ptr->create_non_msg_event(*it, MWS_MSG_BOS, false);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf push_back_non_msg_event MWS_MSG_BOS" << std::endl;
            g_fd_table[*it].rcv_ptr->evq_ptr->push_back_non_msg_event(event_ptr);
            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " create MWS_MSG_BOS" << std::endl;

            // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
            if (g_mws_log_level >= 1)
            {
              std::string log_body_role_addr_info;
              fd_info_log(*it, log_body_role_addr_info);
              std::string log_body = "create MWS_MSG_BOS event. " + log_body_role_addr_info;
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
            // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

            // 將 g_fd_table 的 status 更新為 FD_STATUS_RCV_TOPIC_CHECK_OK.
            update_g_fd_table_status(*it,
                                     FD_STATUS_RCV_TOPIC_CHECK_OK,
                                     __func__,
                                     __LINE__);

            // Begin: 刪除 ctx_list_wait_to_check_topic_rcv_conn_session 中的資料.
            {
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
              //          << " bf ctx_list_wait_to_check_topic_rcv_session.size()"
              //          << std::to_string( g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.size() ) << std::endl;
              int rtv = g_fd_table[*it].rcv_ptr->ctx_ptr->erase_ctx_list_wait_to_check_topic_rcv_session(*it);
              if (rtv != 0)
              {
                // ctx_list_wait_to_check_topic_rcv_session 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(*it) + " does not exist in ctx_list_wait_to_check_topic_rcv_session";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
              //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ )
              //          << " af ctx_list_wait_to_check_topic_rcv_session.size()"
              //          << std::to_string( g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.size() ) << std::endl;
            }
            // End: 刪除 ctx_list_wait_to_check_topic_rcv_conn_session 中的資料.

            //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " !!! rcv check topic ok" << std::endl;
            //sleep(3);
          }
          else
          {
            //std::cout << __func__ << ":" << __LINE__ << " !!! rcv check topic fail" << std::endl;
            //sleep(3);

            // topic name 檢核錯誤.
            // 斷線.
            rcv_topic_check_error(it, __func__, __LINE__);
          }
        }
        // End: 檢核 topic name.
      } // if (accu_rcv_size == g_size_of_mws_topic_name_msg_t)
    }
    else if ((rtv == 0) || (rtv == -1))
    {
      //std::cout << __func__ << ":" << __LINE__ << " step_rcv_wait_topic_name read data fail" << std::endl;
      //sleep(5);

      flag_break_while_loop = true;
      // fd 發生問題, 要斷線.
      rcv_topic_check_error(it, __func__, __LINE__);
    }
    // EAGAIN or EWOULDBLOCK or EINTR.
    else if (rtv == (-2))
    {
      flag_break_while_loop = true;
      // fd 連續發生太多次 EAGAIN/EWOULDBLOCK/EINTR, 要斷線.
      rcv_topic_check_error(it, __func__, __LINE__);
    }

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " af call recv_data" << std::endl;
  } // while (flag_break_while_loop == false)

  return;
}

void rcv_ready_error(std::deque<fd_t>::iterator& it,
                     const std::string function,
                     const int line_no)
{
  std::string topic_name = g_fd_table[*it].rcv_ptr->topic_name;

  if (g_fd_table[*it].status == FD_STATUS_RCV_FD_FAIL)
  {
    std::string log_body_role_addr_info;
    fd_info_log(*it, log_body_role_addr_info);
    std::string log_body = function + " error. " + log_body_role_addr_info;
    write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

    return;
  }

  // Begin: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].rcv_ptr->erase_rcv_list_connected_src_address(g_fd_table[*it].rcv_listen_addr_info);
    if (rtv != 0)
    {
      str_ip_port_t rcv_listen_addr;
      sockaddr_in_t_to_string(g_fd_table[*it].rcv_listen_addr_info,
                              rcv_listen_addr.str_ip,
                              rcv_listen_addr.str_port);

      // rcv_list_connected_src_address 沒有該 sockaddr_in_t 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) +
                 "(" + rcv_listen_addr.str_ip + ":" + rcv_listen_addr.str_port +
                 ") does not exist in rcv_listen_addr_info";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.

  // Begin: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].rcv_ptr->erase_rcv_connect_fds(*it);
    if (rtv != 0)
    {
      // rcv_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in rcv_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.

  // (在 dispatch MWS_MSG_EOS 時移除)
  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  /*{
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[*it].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[*it].rcv_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " evq_list_owned_fds.size():" << std::to_string(g_fd_table[*it].rcv_ptr->evq_ptr->evq_list_owned_fds.size()) << std::endl;
    int rtv = g_fd_table[*it].rcv_ptr->evq_ptr->erase_evq_list_owned_fds(*it);
    if (rtv != 0)
    {
      //std::cout << std::string(__func__) << std::endl;
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[*it].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[*it].rcv_ptr->evq_ptr->evq_unlock();
    #endif
  }*/
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_owned_rcv_fds, 所以要用 it 直接刪除.
    //pthread_mutex_lock(&(g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds_mutex));
    g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds.erase(it);
    //pthread_mutex_unlock(&(g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_owned_rcv_fds_mutex));
  }
  // End: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(*it, &g_fd_table[*it].rcv_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove rcv fd:" +
                           std::to_string(*it) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_FD_FAIL.
  //g_fd_table[*it].status = FD_STATUS_RCV_FD_FAIL;
  update_g_fd_table_status(*it,
                           FD_STATUS_RCV_FD_FAIL,
                           __func__,
                           __LINE__);

  // (在 dispatch MWS_MSG_EOS 時放入) 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
  //g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(*it);

  // Begin: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  {
    wait_to_connect_rcv_session_t temp_obj;
    temp_obj.rcv_ptr = g_fd_table[*it].rcv_ptr;
    temp_obj.rcv_connection_setting = g_fd_table[*it].rcv_connection_setting;
    temp_obj.next_port = g_fd_table[*it].rcv_connection_setting.listen_addr.low_port;
    temp_obj.try_cnt = 0;
    g_fd_table[*it].rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
  }
  // End: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.

  // 產生 MWS_MSG_EOS.
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf create MWS_MSG_EOS" << std::endl;
  mws_event_t* event_ptr = g_fd_table[*it].rcv_ptr->evq_ptr->create_non_msg_event(*it, MWS_MSG_EOS, false);
  g_fd_table[*it].rcv_ptr->evq_ptr->push_back_non_msg_event(event_ptr);
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " create MWS_MSG_EOS" << std::endl;

  // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
  if (g_mws_log_level >= 1)
  {
    std::string log_body_role_addr_info;
    fd_info_log(*it, log_body_role_addr_info);
    std::string log_body = "create MWS_MSG_EOS event. " + log_body_role_addr_info;
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }
  // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

  std::string log_body_role_addr_info;
  fd_info_log(*it, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  //std::cout << __func__ << ":" << __LINE__ << " " << log_body << std::endl;

  return;
}

void step_rcv_ready(std::deque<fd_t>::iterator& it)
{
  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " bf call recv_data" << std::endl;
  mws_msg_evq_buffer_t* mws_msg_evq_buffer_t_ptr = (mws_msg_evq_buffer_t*)(g_fd_table[*it].msg_evq_ptr->push_back_a_black_block(g_fd_table[*it].msg_evq_number));
  //memset(mws_msg_evq_buffer_t_ptr, 0, sizeof(mws_msg_evq_buffer_t));
  ssize_t rtv = recv_data((void*)&(mws_msg_evq_buffer_t_ptr->buffer[g_size_of_mws_pkg_t]),
                          *it,
                          MAX_RECV_SIZE,
                          MAX_NUM_OF_RETRIES_RECV_MSG);
  if (rtv > 0)
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " recv_data rtv > 0" << std::endl;
    //sleep(3);
    mws_msg_evq_buffer_t_ptr->begin_pos = g_size_of_mws_pkg_t;
    mws_msg_evq_buffer_t_ptr->data_size = rtv;
    mws_msg_evq_buffer_t_ptr->filler_1 = 0xEFEEEDEC;
    mws_msg_evq_buffer_t_ptr->filler_2 = 0xEBEAE9E8;
    mws_msg_evq_buffer_t_ptr->filler_3 = 0xE7E6E5E4;

    g_fd_table[*it].rcv_ptr->evq_ptr->flag_must_unlock = true;
  }
  else if ((rtv == 0) || (rtv == -1))
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " step_rcv_ready recv_data error" << std::endl;
    //sleep(5);
    g_fd_table[*it].msg_evq_ptr->pop_back(g_fd_table[*it].msg_evq_number);
    // fd 發生問題, 要斷線.
    rcv_ready_error(it, __func__, __LINE__);
  }
  // EAGAIN or EWOULDBLOCK or EINTR.
  else if (rtv == (-2))
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " recv_data rtv == -2" << std::endl;
    //sleep(3);
    // 刪去新建卻沒有資料的 black block.
    int16_t rtv = g_fd_table[*it].msg_evq_ptr->pop_back(g_fd_table[*it].msg_evq_number);
    if (rtv != 0)
    {
      std::string log_body =
        "fast_deque::pop_back() fail (fd:" + std::to_string(*it) + ", rtv: " + std::to_string(rtv) + ")";
      std::string topic_name = g_fd_table[*it].rcv_ptr->topic_name;
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

      // fd 發生問題, 要斷線.
      rcv_ready_error(it, __func__, __LINE__);
    }
  }

  return;
}

void src_conn_topic_check_error(std::deque<fd_t>::iterator& it,
                                const std::string function,
                                const int line_no)
{
  std::string topic_name = g_fd_table[*it].src_conn_ptr->topic_name;

  // Begin: 移除 src 內 src_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].src_conn_ptr->erase_src_connect_fds(*it);
    if (rtv != 0)
    {
      // src_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in src_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 src 內 src_connect_fds 內該 fd 的資料.

  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  {
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[*it].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->erase_evq_list_owned_fds(*it);
    if (rtv != 0)
    {
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[*it].src_conn_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[*it].src_conn_ptr->evq_ptr->evq_unlock();
    #endif
  }
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_wait_to_check_topic_src_conn_session 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].src_conn_ptr->ctx_ptr->erase_ctx_list_wait_to_check_topic_src_conn_session(*it);
    if (rtv != 0)
    {
      // ctx_list_wait_to_check_topic_src_conn_session 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in ctx_list_wait_to_check_topic_src_conn_session";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_wait_to_check_topic_src_conn_session 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_owned_src_conn_fds, 所以要用 it 直接刪除.
    //pthread_mutex_lock(&(g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
    g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds.erase(it);
    //pthread_mutex_unlock(&(g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
  }
  // End: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(*it, &g_fd_table[*it].src_conn_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove src conn fd:" +
                           std::to_string(*it) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // Begin: 維護 g_fd_table.
  {
    // 修改 g_fd_table 的 status 為 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE.
    //g_fd_table[*it].status = FD_STATUS_SRC_CONN_WAIT_TO_CLOSE;
    update_g_fd_table_status(*it,
                             FD_STATUS_SRC_CONN_WAIT_TO_CLOSE,
                             __func__,
                             __LINE__);
  }
  // End: 維護 g_fd_table.

  // 把 fd 放入 ctx 的 ctx_list_wait_to_close_src_conn_fds.
  g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_wait_to_close_src_conn_fds.push_back(*it);

  std::string log_body_role_addr_info;
  fd_info_log(*it, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

void step_src_conn_wait_fffd(std::deque<fd_t>::iterator& it)
{
  bool flag_break_while_loop = false;
  while (flag_break_while_loop == false)
  {
    char rcv_buff[1];
    ssize_t rtv = recv_data((void*)&rcv_buff[0], *it, 1, MAX_NUM_OF_RETRIES_RECV_FFFD_FEFC);
    if (rtv > 0)
    {
      if (rcv_buff[0] == (char)0xFF)
      {
        // 將 g_fd_table 的 status 改為 FD_STATUS_SRC_CONN_WAIT_FD.
        //g_fd_table[*it].status = FD_STATUS_SRC_CONN_WAIT_FD;
        update_g_fd_table_status(*it,
                                 FD_STATUS_SRC_CONN_WAIT_FD,
                                 __func__,
                                 __LINE__);

        // Begin: send FC to rcv.
        {
          // 只能送一次 0xFC.
          if (g_fd_table[*it].src_conn_sent_FC == false)
          {
            // 傳送 0xFC.
            char send_buff[1];
            send_buff[0] = (char)0xFC;
            ssize_t rtv = send_topic_check_code((void*)&send_buff[0], *it, 1);
            if (rtv < 0)
            {
              // fd 發生問題, 要斷線.
              src_conn_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].src_conn_sent_FC = true;
            }
          }
        }
        // End: send FC to rcv.
      }
      else if (rcv_buff[0] == (char)0xFD)
      {
        flag_break_while_loop = true;

        // 將 g_fd_table 的 status 改為 FD_STATUS_SRC_CONN_WAIT_TOPIC_NAME.
        //g_fd_table[*it].status = FD_STATUS_SRC_CONN_WAIT_TOPIC_NAME;
        update_g_fd_table_status(*it,
                                 FD_STATUS_SRC_CONN_WAIT_TOPIC_NAME,
                                 __func__,
                                 __LINE__);

        // Begin: send FC to rcv.
        {
          // 只能送一次 0xFC.
          if (g_fd_table[*it].src_conn_sent_FC == false)
          {
            // 傳送 0xFC.
            char send_buff[1];
            send_buff[0] = (char)0xFC;
            ssize_t rtv = send_topic_check_code((void*)&send_buff[0], *it, 1);
            if (rtv < 0)
            {
              // fd 發生問題, 要斷線.
              src_conn_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].src_conn_sent_FC = true;
            }
          }
        }
        // End: send FC to rcv.

        // Begin: send topic name to rcv.
        {
          // 只能送一次 topic name.
          if (g_fd_table[*it].src_conn_sent_topic_name == false)
          {
            // 傳送 topic name
            mws_topic_name_msg_t temp_obj;
            memset(&temp_obj, 0, g_size_of_mws_topic_name_msg_t);
            temp_obj.topic_name_len = htons((uint16_t)g_fd_table[*it].src_conn_ptr->topic_name.size());
            memcpy((void*)(&temp_obj.topic_name[0]),
                   (void*)(g_fd_table[*it].src_conn_ptr->topic_name.c_str()),
                   g_fd_table[*it].src_conn_ptr->topic_name.size());
            ssize_t rtv = send_topic_name(temp_obj, *it);
            if (rtv < 0)
            {
              // fd 發生問題, 要斷線.
              src_conn_topic_check_error(it, __func__, __LINE__);
            }
            else
            {
              g_fd_table[*it].src_conn_sent_topic_name = true;
            }
          }
        }
        // End: send topic name to rcv.
      }
    }
    else if ((rtv == 0) || (rtv == -1))
    {
      flag_break_while_loop = true;
      // fd 發生問題, 要斷線.
      src_conn_topic_check_error(it, __func__, __LINE__);
    }
    // EAGAIN or EWOULDBLOCK or EINTR.
    else if (rtv == (-2))
    {
      flag_break_while_loop = true;
    }
  } // while (flag_break_while_loop == false)

  return;
}

void step_src_conn_wait_topic_name(std::deque<fd_t>::iterator& it)
{
  std::string topic_name = g_fd_table[*it].src_conn_ptr->topic_name;

  // 累計接收到的 byte 數.
  size_t accu_rcv_size = 0;
  // 接收資料用的 buffer.
  //char rcv_buff[g_size_of_mws_topic_name_msg_t]; // 無法這樣宣告.
  char rcv_buff[MAX_MSG_SIZE];

  bool flag_break_while_loop = false;
  while (flag_break_while_loop == false)
  {
    ssize_t rtv = recv_data((void*)&rcv_buff[accu_rcv_size],
                            *it,
                            (g_size_of_mws_topic_name_msg_t - accu_rcv_size),
                            MAX_NUM_OF_RETRIES_RECV_TOPIC_NAME);
    if (rtv > 0)
    {
      accu_rcv_size += rtv;
      if (accu_rcv_size == g_size_of_mws_topic_name_msg_t)
      {
        flag_break_while_loop = true;
        // Begin: 檢核 topic name.
        {
          mws_topic_name_msg_t temp_obj;
          memcpy((void*)&temp_obj, &rcv_buff[0], g_size_of_mws_topic_name_msg_t);
          std::string rcv_topic_name(&temp_obj.topic_name[0], ntohs(temp_obj.topic_name_len));

          if (rcv_topic_name == g_fd_table[*it].src_conn_ptr->topic_name)
          {
            // topic name 檢核正確.
            // 移除 ctx 內 ctx_list_wait_to_check_topic_src_conn_session 內該 fd 的資料.
            // 產生 MWS_SRC_EVENT_CONNECT.
            mws_event_t* event_ptr = g_fd_table[*it].src_conn_ptr->evq_ptr->create_non_msg_event(*it, MWS_SRC_EVENT_CONNECT, false);
            g_fd_table[*it].src_conn_ptr->evq_ptr->push_back_non_msg_event(event_ptr);

            // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
            if (g_mws_log_level >= 1)
            {
              std::string log_body_role_addr_info;
              fd_info_log(*it, log_body_role_addr_info);
              std::string log_body = "create MWS_SRC_EVENT_CONNECT event. " + log_body_role_addr_info;
              write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
            }
            // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

            // 將 g_fd_table 的 status 更新為 FD_STATUS_SRC_CONN_TOPIC_CHECK_OK.
            update_g_fd_table_status(*it,
                                     FD_STATUS_SRC_CONN_TOPIC_CHECK_OK,
                                     __func__,
                                     __LINE__);

            // Begin: 刪除 ctx_list_wait_to_check_topic_src_conn_session 中的資料.
            {
              int rtv = g_fd_table[*it].src_conn_ptr->ctx_ptr->erase_ctx_list_wait_to_check_topic_src_conn_session(*it);
              if (rtv != 0)
              {
                // ctx_list_wait_to_check_topic_src_conn_session 沒有該 fd 資料, 刷錯誤訊息.
                std::string log_body;
                log_body = "fd: " + std::to_string(*it) + " does not exist in ctx_list_wait_to_check_topic_src_conn_session";
                write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
              }
            }
            // End: 刪除 ctx_list_wait_to_check_topic_src_conn_session 中的資料.
          }
          else
          {
            // topic name 檢核錯誤, 斷線.
            src_conn_topic_check_error(it, __func__, __LINE__);
          }
        }
        // End: 檢核 topic name.
      } // if (accu_rcv_size == g_size_of_mws_topic_name_msg_t)
    }
    else if ((rtv == 0) || (rtv == -1))
    {
      flag_break_while_loop = true;
      // fd 發生問題, 要斷線.
      src_conn_topic_check_error(it, __func__, __LINE__);
    }
    // EAGAIN or EWOULDBLOCK or EINTR.
    else if (rtv == (-2))
    {
      flag_break_while_loop = true;
      // fd 連續發生太多次 EAGAIN/EWOULDBLOCK/EINTR, 要斷線.
      src_conn_topic_check_error(it, __func__, __LINE__);
    }
  } // while (flag_break_while_loop == false)

  return;
}

void src_conn_ready_error(std::deque<fd_t>::iterator& it,
                          const std::string function,
                          const int line_no)
{
  std::string topic_name = g_fd_table[*it].src_conn_ptr->topic_name;

  if (g_fd_table[*it].status == FD_STATUS_SRC_CONN_FD_FAIL)
  {
    std::string log_body_role_addr_info;
    fd_info_log(*it, log_body_role_addr_info);
    std::string log_body = function + " error. " + log_body_role_addr_info;
    write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

    return;
  }

  // Begin: 移除 src 內 src_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[*it].src_conn_ptr->erase_src_connect_fds(*it);
    if (rtv != 0)
    {
      // src_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in src_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 src 內 src_connect_fds 內該 fd 的資料.

  // (在 dispatch MWS_SRC_EVENT_DISCONNECT 時移除).
  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  /*{
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[*it].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[*it].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[*it].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[*it].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    int rtv = g_fd_table[*it].src_conn_ptr->evq_ptr->erase_evq_list_owned_fds(*it);
    if (rtv != 0)
    {
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(*it) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[*it].src_conn_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[*it].src_conn_ptr->evq_ptr->evq_unlock();
    #endif
  }*/
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_owned_src_conn_fds, 所以要用 it 直接刪除.
    //pthread_mutex_lock(&(g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
    g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds.erase(it);
    //pthread_mutex_unlock(&(g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_owned_src_conn_fds_mutex));
  }
  // End: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(*it, &g_fd_table[*it].src_conn_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove src conn fd:" +
                           std::to_string(*it) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_SRC_CONN_FD_FAIL.
  //g_fd_table[*it].status = FD_STATUS_SRC_CONN_FD_FAIL;
  update_g_fd_table_status(*it,
                           FD_STATUS_SRC_CONN_FD_FAIL,
                           __func__,
                           __LINE__);

  // (在 dispatch MWS_SRC_EVENT_DISCONNECT 時放入) 把 fd 放入 ctx 的 ctx_list_wait_to_close_src_conn_fds.
  //g_fd_table[*it].src_conn_ptr->ctx_ptr->ctx_list_wait_to_close_src_conn_fds.push_back(*it);

  // 產生 MWS_SRC_EVENT_DISCONNECT.
  mws_event_t* event_ptr = g_fd_table[*it].src_conn_ptr->evq_ptr->create_non_msg_event(*it, MWS_SRC_EVENT_DISCONNECT, false);
  g_fd_table[*it].src_conn_ptr->evq_ptr->push_back_non_msg_event(event_ptr);

  // Begin: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.
  if (g_mws_log_level >= 1)
  {
    std::string log_body_role_addr_info;
    fd_info_log(*it, log_body_role_addr_info);
    std::string log_body = "create MWS_SRC_EVENT_DISCONNECT event. " + log_body_role_addr_info;
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }
  // End: g_mws_log_level >= 1 時, 刷 topic name 檢核正確的 log.

  std::string log_body_role_addr_info;
  fd_info_log(*it, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

void step_src_conn_ready(std::deque<fd_t>::iterator& it)
{
  mws_msg_evq_buffer_t* mws_msg_evq_buffer_t_ptr = (mws_msg_evq_buffer_t*)(g_fd_table[*it].msg_evq_ptr->push_back_a_black_block(g_fd_table[*it].msg_evq_number));
  //memset(mws_msg_evq_buffer_t_ptr, 0, sizeof(mws_msg_evq_buffer_t));
  ssize_t rtv = recv_data((void*)&(mws_msg_evq_buffer_t_ptr->buffer[g_size_of_mws_pkg_t]),
                          *it,
                          MAX_RECV_SIZE,
                          MAX_NUM_OF_RETRIES_RECV_MSG);
  if (rtv > 0)
  {
    mws_msg_evq_buffer_t_ptr->begin_pos = g_size_of_mws_pkg_t;
    mws_msg_evq_buffer_t_ptr->data_size = rtv;
    mws_msg_evq_buffer_t_ptr->filler_1 = 0xEFEEEDEC;
    mws_msg_evq_buffer_t_ptr->filler_2 = 0xEBEAE9E8;
    mws_msg_evq_buffer_t_ptr->filler_3 = 0xE7E6E5E4;

    g_fd_table[*it].src_conn_ptr->evq_ptr->flag_must_unlock = true;
  }
  else if ((rtv == 0) || (rtv == -1))
  {
    g_fd_table[*it].msg_evq_ptr->pop_back(g_fd_table[*it].msg_evq_number);
    // fd 發生問題, 要斷線.
    src_conn_ready_error(it, __func__, __LINE__);
  }
  // EAGAIN or EWOULDBLOCK or EINTR.
  else if (rtv == (-2))
  {
    // 刪去新建卻沒有資料的 black block.
    int16_t rtv = g_fd_table[*it].msg_evq_ptr->pop_back(g_fd_table[*it].msg_evq_number);
    if (rtv != 0)
    {
      std::string log_body =
        "fast_deque::pop_back() fail (fd:" + std::to_string(*it) + ", rtv: " + std::to_string(rtv) + ")";
      std::string topic_name = g_fd_table[*it].src_conn_ptr->topic_name;
      write_to_log(topic_name, -1, "E", __FILE__, __func__, __LINE__, log_body);

      // fd 發生問題, 要斷線.
      src_conn_ready_error(it, __func__, __LINE__);
    }
  }

  return;
}

void update_g_fd_table_status(const fd_t fd,
                              const int16_t new_status,
                              const std::string function,
                              const int line_no)
{
  switch (g_fd_table[fd].role)
  {
    case FD_ROLE_SRC_LISTEN:
    {
      if ((new_status >= g_fd_table[fd].status) && (new_status < 20))
      {
        g_fd_table[fd].status = new_status;
      }
      else if ((new_status == FD_STATUS_UNKNOWN) &&
               (g_fd_table[fd].status == FD_STATUS_SRC_LISTEN_WAIT_TO_CLOSE))
      {
        g_fd_table[fd].status = new_status;
      }
      else
      {
        std::string log_body_role_addr_info;
        fd_info_log(fd, log_body_role_addr_info);
        std::string log_body = "FD_ROLE_SRC_LISTEN update_fd_status from " + std::to_string(g_fd_table[fd].status) +
                               " to " + std::to_string(new_status) +
                               " error. " + log_body_role_addr_info;
        std::string topic_name = g_fd_table[fd].src_listen_ptr->topic_name;
        write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);
      }

      break;
    }
    case FD_ROLE_SRC_CONN:
    {
      if ((new_status >= g_fd_table[fd].status) && (new_status < 30))
      {
        g_fd_table[fd].status = new_status;
      }
      else if ((new_status == FD_STATUS_UNKNOWN) &&
               (g_fd_table[fd].status == FD_STATUS_SRC_CONN_WAIT_TO_CLOSE))
      {
        g_fd_table[fd].status = new_status;
      }
      else
      {
        std::string log_body_role_addr_info;
        fd_info_log(fd, log_body_role_addr_info);
        std::string log_body = "FD_ROLE_SRC_CONN update_fd_status from " + std::to_string(g_fd_table[fd].status) +
                               " to " + std::to_string(new_status) +
                               " error. " + log_body_role_addr_info;
        std::string topic_name = g_fd_table[fd].src_conn_ptr->topic_name;
        write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);
      }

      break;
    }
    case FD_ROLE_RCV:
    {
      if ((new_status >= g_fd_table[fd].status) && (new_status < 40))
      {
        g_fd_table[fd].status = new_status;
      }
      else if ((new_status == FD_STATUS_UNKNOWN) &&
               (g_fd_table[fd].status == FD_STATUS_RCV_WAIT_TO_CLOSE))
      {
        g_fd_table[fd].status = new_status;
      }
      else
      {
        std::string log_body_role_addr_info;
        fd_info_log(fd, log_body_role_addr_info);
        std::string log_body = "FD_ROLE_RCV update_fd_status from " + std::to_string(g_fd_table[fd].status) +
                               " to " + std::to_string(new_status) +
                               " error. " + log_body_role_addr_info;
        std::string topic_name = g_fd_table[fd].rcv_ptr->topic_name;
        write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);
      }

      break;
    }
    case FD_ROLE_UNKNOWN:
    {
      if (new_status == FD_STATUS_UNKNOWN)
      {
        g_fd_table[fd].status = new_status;
      }
      else
      {
        std::string log_body_role_addr_info;
        fd_info_log(fd, log_body_role_addr_info);
        std::string log_body = "FD_ROLE_UNKNOWN update_fd_status from " + std::to_string(g_fd_table[fd].status) +
                               " to " + std::to_string(new_status) +
                               " error. " + log_body_role_addr_info;
        write_to_log("", -1, "E", __FILE__, function, line_no, log_body);
      }

      break;
    }
    default:
    {
      std::string log_body_role_addr_info;
      fd_info_log(fd, log_body_role_addr_info);
      std::string log_body = "Role " + std::to_string(g_fd_table[fd].role) +
                             " update_fd_status from " + std::to_string(g_fd_table[fd].status) +
                             " to " + std::to_string(new_status) +
                             " error. " + log_body_role_addr_info;
      write_to_log("", -1, "E", __FILE__, function, line_no, log_body);

      break;
    }
  }

  return;
}

void step_send_fe_error(std::deque<wait_to_check_topic_src_conn_session_t>::iterator& it,
                        const std::string function,
                        const int line_no)
{
  std::string topic_name = g_fd_table[it->fd].src_conn_ptr->topic_name;

  // Begin: 移除 src 內 src_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[it->fd].src_conn_ptr->erase_src_connect_fds(it->fd);
    if (rtv != 0)
    {
      // src_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in src_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 src 內 src_connect_fds 內該 fd 的資料.

  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  {
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[it->fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[it->fd].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[it->fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[it->fd].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    int rtv = g_fd_table[it->fd].src_conn_ptr->evq_ptr->erase_evq_list_owned_fds(it->fd);
    if (rtv != 0)
    {
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[it->fd].src_conn_ptr->evq_ptr->evq_unlock();
    #endif
  }
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_wait_to_check_topic_src_conn_session 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_wait_to_check_topic_src_conn_session, 所以要用 it 直接刪除.
    g_fd_table[it->fd].src_conn_ptr->ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.erase(it);
  }
  // End: 移除 ctx 內 ctx_list_wait_to_check_topic_src_conn_session 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[it->fd].src_conn_ptr->ctx_ptr->erase_ctx_list_owned_src_conn_fds(it->fd);
    if (rtv != 0)
    {
      // ctx_list_owned_src_conn_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in ctx_list_owned_src_conn_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_owned_src_conn_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(it->fd, &g_fd_table[it->fd].src_conn_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove src conn fd:" +
                           std::to_string(it->fd) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE.
  //g_fd_table[*it].status = FD_STATUS_SRC_CONN_WAIT_TO_CLOSE;
  update_g_fd_table_status(it->fd,
                           FD_STATUS_SRC_CONN_WAIT_TO_CLOSE,
                           __func__,
                           __LINE__);

  // 把 fd 放入 ctx 的 ctx_list_wait_to_close_src_conn_fds.
  g_fd_table[it->fd].src_conn_ptr->ctx_ptr->ctx_list_wait_to_close_src_conn_fds.push_back(it->fd);

  std::string log_body_role_addr_info;
  fd_info_log(it->fd, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

void step_send_ff_error(std::deque<wait_to_check_topic_rcv_session_t>::iterator& it,
                        const std::string function,
                        const int line_no)
{
  std::string topic_name = g_fd_table[it->fd].rcv_ptr->topic_name;

  // Begin: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.
  {
    int rtv = g_fd_table[it->fd].rcv_ptr->erase_rcv_list_connected_src_address(g_fd_table[it->fd].rcv_listen_addr_info);
    if (rtv != 0)
    {
      str_ip_port_t rcv_listen_addr;
      sockaddr_in_t_to_string(g_fd_table[it->fd].rcv_listen_addr_info,
                              rcv_listen_addr.str_ip,
                              rcv_listen_addr.str_port);

      // rcv_list_connected_src_address 沒有該 sockaddr_in_t 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) +
                 "(" + rcv_listen_addr.str_ip + ":" + rcv_listen_addr.str_port +
                 ") does not exist in rcv_listen_addr_info";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_list_connected_src_address 內該 fd 的資料.

  // Begin: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[it->fd].rcv_ptr->erase_rcv_connect_fds(it->fd);
    if (rtv != 0)
    {
      // rcv_connect_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in rcv_connect_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.

  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  {
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[it->fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[it->fd].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[it->fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[it->fd].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    int rtv = g_fd_table[it->fd].rcv_ptr->evq_ptr->erase_evq_list_owned_fds(it->fd);
    if (rtv != 0)
    {
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[it->fd].rcv_ptr->evq_ptr->evq_unlock();
    #endif
  }
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.
  {
    // 由於 while 大迴圈使用 ctx_list_wait_to_check_topic_rcv_session, 所以要用 it 直接刪除.
    g_fd_table[it->fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.erase(it);
  }
  // End: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[it->fd].rcv_ptr->ctx_ptr->erase_ctx_list_owned_rcv_fds(it->fd);
    if (rtv != 0)
    {
      // ctx_list_owned_rcv_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(it->fd) + " does not exist in ctx_list_owned_rcv_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(it->fd, &g_fd_table[it->fd].rcv_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove rcv fd:" +
                           std::to_string(it->fd) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_WAIT_TO_CLOSE.
  //g_fd_table[*it].status = FD_STATUS_RCV_WAIT_TO_CLOSE;
  update_g_fd_table_status(it->fd,
                           FD_STATUS_RCV_WAIT_TO_CLOSE,
                           __func__,
                           __LINE__);

  // 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
  g_fd_table[it->fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(it->fd);
  if (g_mws_log_level >= 1)
  {
    std::string log_body = "push to ctx_list_wait_to_close_rcv_fds-rcv fd: " + std::to_string(it->fd);
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }

  // Begin: 將 rcv 要做 reconnect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  {
    wait_to_connect_rcv_session_t temp_obj;
    temp_obj.rcv_ptr = g_fd_table[it->fd].rcv_ptr;
    temp_obj.rcv_connection_setting = g_fd_table[it->fd].rcv_connection_setting;
    temp_obj.next_port = g_fd_table[it->fd].rcv_connection_setting.listen_addr.low_port;
    temp_obj.try_cnt = 0;
    g_fd_table[it->fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
  }
  // End: 將 rcv 要做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.

  std::string log_body_role_addr_info;
  fd_info_log(it->fd, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}

void step_rcv_connect(std::deque<wait_to_connect_rcv_session_t>::iterator& it,
                      mws_ctx_t* ctx_ptr,
                      const std::string function,
                      const int line_no)
{
  sockaddr_in_t rcv_listen_addr_info;
  sockaddr_in_t rcv_addr_info;
  int rcv_fd = create_connect_socket(*it,
                                     rcv_listen_addr_info,
                                     rcv_addr_info);

  // 正確連線到 src, 更新 ctx_list_wait_to_check_topic_rcv_session,
  // 並刪除 ctx_list_wait_to_connect_rcv_session 中的設定.
  if (rcv_fd > 0)
  {
    #if (MWS_DEBUG == 1)
      it->rcv_ptr->evq_ptr->evq_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      it->rcv_ptr->evq_ptr->evq_lock();
    #endif

    #if (MWS_DEBUG == 1)
      g_fd_table[rcv_fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[rcv_fd].fd_lock();
    #endif

    // Begin: 將要 check topic 的 session 新增到 ctx_list_wait_to_check_topic_rcv_session.
    {
      wait_to_check_topic_rcv_session_t temp_obj;
      temp_obj.fd = rcv_fd;
      temp_obj.rcv_ptr = it->rcv_ptr;
      temp_obj.rcv_listen_addr_info = rcv_listen_addr_info;
      temp_obj.rcv_addr_info = rcv_addr_info;
      ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.push_back(temp_obj);
    }
    // End: 將要 check topic 的 session 新增到 ctx_list_wait_to_check_topic_rcv_session.

    // Begin: 更新 ctx_ptr->ctx_list_owned_rcv_fds.
    //pthread_mutex_lock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
    #if (MWS_DEBUG == 1)
      ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      ctx_ptr->ctx_list_owned_rcv_fds_mutex_lock();
    #endif

    ctx_ptr->ctx_list_owned_rcv_fds.push_back(rcv_fd);

    //pthread_mutex_unlock(&(ctx_ptr->ctx_list_owned_rcv_fds_mutex));
    #if (MWS_DEBUG == 1)
      ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      ctx_ptr->ctx_list_owned_rcv_fds_mutex_unlock();
    #endif
    // End: 更新 ctx_ptr->ctx_list_owned_rcv_fds.

    // 新增 rcv_fd 至 all_set.
    FD_SET(rcv_fd, &ctx_ptr->all_set);
    {
      std::string log_body = "Add rcv fd:" +
                             std::to_string(rcv_fd) +
                             " into all_set ";
      std::string topic_name = it->rcv_ptr->topic_name;
      write_to_log(topic_name, 0, "N", __FILE__, function, line_no, log_body);
    }

    // 更新 ctx::max_fd
    ctx_ptr->update_max_fd(rcv_fd);

    // Begin: 更新 g_fd_table.
    {
      g_fd_table[rcv_fd].fd = rcv_fd;
      g_fd_table[rcv_fd].role = FD_ROLE_RCV;
      update_g_fd_table_status(rcv_fd,
                               FD_STATUS_RCV_PREPARE,
                               __func__,
                               __LINE__);
      g_fd_table[rcv_fd].rcv_ptr = it->rcv_ptr;
      g_fd_table[rcv_fd].rcv_listen_addr_info = rcv_listen_addr_info;
      g_fd_table[rcv_fd].rcv_addr_info = rcv_addr_info;
      g_fd_table[rcv_fd].rcv_sent_FD = false;
      g_fd_table[rcv_fd].rcv_sent_topic_name = false;
      g_fd_table[rcv_fd].rcv_connection_setting = it->rcv_connection_setting;
      g_fd_table[rcv_fd].msg_evq_ptr = new mws_fast_deque_t(sizeof(mws_msg_evq_buffer_t), RCV_NUM_OF_DEFAULT_RECV_BUFFER_BLOCK, RCV_NUM_OF_EXT_RECV_BUFFER_BLOCK);
      g_fd_table[rcv_fd].msg_evq_number = g_fd_table[rcv_fd].msg_evq_ptr->get_new_deque();
      g_fd_table[rcv_fd].send_buffer_ptr = new mws_fast_deque_t(sizeof(mws_send_buff_t), RCV_NUM_OF_DEFAULT_SND_BUFFER_BLOCK, RCV_NUM_OF_EXT_SND_BUFFER_BLOCK);
      g_fd_table[rcv_fd].send_buffer_number = g_fd_table[rcv_fd].send_buffer_ptr->get_new_deque();
    }
    // End: 更新 g_fd_table.

    // Begin: 更新 evq_list_owned_fds.
    {
      g_fd_table[rcv_fd].rcv_ptr->evq_ptr->evq_list_owned_fds.push_back(rcv_fd);
    }
    // End: 更新 evq_list_owned_fds.

    // 從 ctx_list_wait_to_connect_rcv_session.erase 刪除已經完成連線的 rcv 設定.
    it = ctx_ptr->ctx_list_wait_to_connect_rcv_session.erase(it);

    #if (MWS_DEBUG == 1)
      g_fd_table[rcv_fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[rcv_fd].fd_unlock();
    #endif

    #if (MWS_DEBUG == 1)
      g_fd_table[rcv_fd].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[rcv_fd].rcv_ptr->evq_ptr->evq_unlock();
    #endif
  } // if (rcv_fd > 0)
  else
  {
    ++it->try_cnt;
    // 換下一組設定.
    if (it != ctx_ptr->ctx_list_wait_to_connect_rcv_session.end())
    {
      ++it;
    }
  } // else of if (rcv_fd > 0)

  return;
}

void rcv_topic_check_timeout_error(fd_t fd,
                                   const std::string function,
                                   const int line_no)
{
  std::string topic_name = g_fd_table[fd].rcv_ptr->topic_name;

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
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
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
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 rcv 內 rcv_connect_fds 內該 fd 的資料.

  // Begin: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.
  {
    // Begin: lock evq.
    {
      // try lock evq.
      #if (MWS_DEBUG == 1)
        int rtv = g_fd_table[fd].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
      #else
        int rtv = g_fd_table[fd].rcv_ptr->evq_ptr->evq_trylock();
      #endif

      while (rtv == EBUSY)
      {
        // 解除 fd 的 lock.
        #if (MWS_DEBUG == 1)
          g_fd_table[fd].fd_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          g_fd_table[fd].fd_unlock();
        #endif

        usleep(10);

        // try lock evq.
        #if (MWS_DEBUG == 1)
          rtv = g_fd_table[fd].rcv_ptr->evq_ptr->evq_trylock(std::string(__FILE__), std::string(__func__), int(__LINE__));
        #else
          rtv = g_fd_table[fd].rcv_ptr->evq_ptr->evq_trylock();
        #endif

        if (rtv == 0)
        {
          // lock fd.
          #if (MWS_DEBUG == 1)
            g_fd_table[fd].fd_lock(std::string(__FILE__), std::string(__func__), int(__LINE__));
          #else
            g_fd_table[fd].fd_lock();
          #endif
        }
      }
    }
    // End: lock evq.

    int rtv = g_fd_table[fd].rcv_ptr->evq_ptr->erase_evq_list_owned_fds(fd);
    if (rtv != 0)
    {
      // evq_list_owned_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
    else
    {
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " has been removed from evq_list_owned_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }

    #if (MWS_DEBUG == 1)
      g_fd_table[fd].rcv_ptr->evq_ptr->evq_unlock(std::string(__FILE__), std::string(__func__), int(__LINE__));
    #else
      g_fd_table[fd].rcv_ptr->evq_ptr->evq_unlock();
    #endif
  }
  // End: 移除 mws_evq::evq_list_owned_fds 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].rcv_ptr->ctx_ptr->erase_ctx_list_wait_to_check_topic_rcv_session(fd);
    if (rtv != 0)
    {
      // ctx_list_wait_to_check_topic_rcv_session 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_wait_to_check_topic_rcv_session";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_wait_to_check_topic_rcv_session 內該 fd 的資料.

  // Begin: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.
  {
    int rtv = g_fd_table[fd].rcv_ptr->ctx_ptr->erase_ctx_list_owned_rcv_fds(fd);
    if (rtv != 0)
    {
      // ctx_list_owned_rcv_fds 沒有該 fd 資料, 刷錯誤訊息.
      std::string log_body;
      log_body = "fd: " + std::to_string(fd) + " does not exist in ctx_list_owned_rcv_fds";
      write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 移除 ctx 內 ctx_list_owned_rcv_fds 內該 fd 的資料.

  // 把 fd 從 all_set 中移除.
  FD_CLR(fd, &g_fd_table[fd].rcv_ptr->ctx_ptr->all_set);
  {
    std::string log_body = "Remove rcv fd:" +
                           std::to_string(fd) +
                           " from all_set ";
    write_to_log(topic_name, 0, "N", __FILE__, __func__, __LINE__, log_body);
  }

  // 修改 g_fd_table 的 status 為 FD_STATUS_RCV_WAIT_TO_CLOSE.
  //g_fd_table[fd].status = FD_STATUS_RCV_WAIT_TO_CLOSE;
  update_g_fd_table_status(fd,
                           FD_STATUS_RCV_WAIT_TO_CLOSE,
                           __func__,
                           __LINE__);

  // Begin: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.
  {
    // 取得下次連線用的 port.
    uint16_t next_port = ntohs(g_fd_table[fd].rcv_listen_addr_info.sin_port);
    update_address_port(next_port, g_fd_table[fd].rcv_connection_setting.listen_addr);

    wait_to_connect_rcv_session_t temp_obj;
    temp_obj.rcv_ptr = g_fd_table[fd].rcv_ptr;
    temp_obj.rcv_connection_setting = g_fd_table[fd].rcv_connection_setting;
    temp_obj.next_port = next_port;
    temp_obj.try_cnt = 0;
    g_fd_table[fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_connect_rcv_session.push_back(temp_obj);
  }
  // End: 將 rcv 要重新做 connect 的設定放入 ctx_list_wait_to_connect_rcv_session.

  // 把 fd 放入 ctx 的 ctx_list_wait_to_close_rcv_fds.
  g_fd_table[fd].rcv_ptr->ctx_ptr->ctx_list_wait_to_close_rcv_fds.push_back(fd);
  if (g_mws_log_level >= 1)
  {
    std::string log_body = "push to ctx_list_wait_to_close_rcv_fds-rcv fd: " + std::to_string(fd);
    write_to_log(topic_name, 99, "D", __FILE__, __func__, __LINE__, log_body);
  }

  std::string log_body_role_addr_info;
  fd_info_log(fd, log_body_role_addr_info);
  std::string log_body = function + " error. " + log_body_role_addr_info;
  write_to_log(topic_name, -1, "E", __FILE__, function, line_no, log_body);

  return;
}
