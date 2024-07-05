//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_SOCKET_CPP 1

#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_type_definition.h"

// debug
#include <iostream>

using namespace mws_log;

inline int mws_socket(int domain,
                      int type,
                      int protocol)
{
  return socket(domain, type, protocol);
}

inline int mws_bind(fd_t sockfd,
                    sockaddr_t* addr_ptr,
                    socklen_t addrlen)
{
  return bind(sockfd, addr_ptr, addrlen);
}

inline int mws_listen(fd_t fd,
                      int backlog)
{
  // The backlog argument defines the maximum length to which the
  // queue of pending connections for sockfd may grow.  If a
  // connection request arrives when the queue is full, the client may
  // receive an error with an indication of ECONNREFUSED or, if the
  // underlying protocol supports retransmission, the request may be
  // ignored so that a later reattempt at connection succeeds.
  return listen(fd, backlog);
}

inline int mws_connect(fd_t sockfd,
                       sockaddr_t* addr_ptr,
                       socklen_t addrlen)
{
  return connect(sockfd, addr_ptr, addrlen);
}

inline int mws_accept(fd_t sockfd,
                      sockaddr_t* addr_ptr,
                      socklen_t* addrlen_ptr)
{
  return accept(sockfd, addr_ptr, addrlen_ptr);
}

inline int mws_select(int nfds,
                      fd_set* rset_ptr,
                      fd_set* writefds_ptr,
                      fd_set* exceptfds_ptr,
                      timeval_t* select_timeout_ptr)
{
  return select(nfds, rset_ptr, writefds_ptr, exceptfds_ptr, select_timeout_ptr);
}

inline ssize_t mws_send(fd_t fd,
                        void* buf_ptr,
                        size_t len,
                        int flags)
{
  return send(fd, buf_ptr, len, flags);
}

// 回傳值有三種可能:
//   1. 同參數 len: 表示所有資料都傳送完畢.
//   2. len > 回傳值 > 0: 表示有部分資料傳送完畢,
//      但遇到連續 EAGAIN/EWOULDBLOCK/EINTR 次數
//      超過 MAX_NUM_OF_RETRIES_SEND.
//   3. -1: 表示 socket 出現問題.
ssize_t mws_send_nonblock(fd_t fd,
                          char* buf_ptr,
                          size_t len,
                          int flags)
{
  // 資料傳送方式:
  // 1. 在 O_NONBLOCK 或 MSG_DONTWAIT 模式下, 嘗試傳送資料.
  // 2. 資料有部分或全部傳送 (rtv > 0):
  //    2.1. cumulative_number_of_bytes_sent 會累計傳送的 byte 數,
  //         並把 try_cnt 清 0.
  //    2.2. 依資料是否傳送完畢分成兩種狀況:
  //       2.2.1  資料沒有全部傳送完畢: 重新再傳送資料 (回到步驟 1).
  //       2.2.2. 資料全部傳送完畢:
  //              到步驟 5 (cumulative_number_of_bytes_sent 值等於 len).
  // 3. 發生 EAGAIN/EWOULDBLOCK/EINTR:
  //    3.1. 累計 try_cnt 次數,
  //    3.2. 依 try_cnt 值有兩種狀況:
  //       3.2.1. try_cnt <= MAX_NUM_OF_RETRIES_SEND:
  //              休息 1 ms, 重新再傳送資料 (回到步驟 1).
  //       3.2.2. try_cnt > MAX_NUM_OF_RETRIES_SEND:
  //              則停止傳送資料,
  //              到步驟 5 (cumulative_number_of_bytes_sent 值小於 len).
  // 4. 如果發生"非" EAGAIN/EWOULDBLOCK/EINTR 錯誤, 回傳 -1.
  // 5. 回傳 cumulative_number_of_bytes_sent.

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
  size_t cumulative_number_of_bytes_sent = 0;
  int32_t try_cnt = 0;
  do
  {
    rtv = send(fd, buf_ptr, (len - cumulative_number_of_bytes_sent), flags);
    if (rtv > 0)
    {
      cumulative_number_of_bytes_sent += rtv;
      buf_ptr += rtv;

      try_cnt = 0;
    }
    else
    {
      // 該連線有問題.
      if (!((errno == EAGAIN) || (errno == EWOULDBLOCK) || (errno == EINTR)))
      {
        std::string log_body =
          "mws_send_nonblock() fail (fd:" + std::to_string(fd) +
          "), errno: " + std::to_string(errno) +
          "(" + strerror(errno) + ")";
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        rtv = (-1);
        break; // while (cumulative_number_of_bytes_sent < len);
      }
      else
      {
        ++try_cnt;

        if (try_cnt <= MAX_NUM_OF_RETRIES_SEND)
        {
          //std::string log_body =
          //  "mws_send_nonblock() is blocked (fd:" + std::to_string(fd) +
          //  "), errno: " + std::to_string(errno) +
          //  "(" + strerror(errno) + ")";
          //write_to_log("", 1, "W", __FILE__, __func__, __LINE__, log_body);

          usleep(1000);
        }
        else
        {
          std::string log_body =
            "mws_send_nonblock() has been blocked more than "+
            std::to_string(MAX_NUM_OF_RETRIES_SEND) +
            " times (fd:" + std::to_string(fd) +
            "), errno: " + std::to_string(errno) +
            "(" + strerror(errno) + ")";
          write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

          rtv = (-2);

          break; // while (cumulative_number_of_bytes_sent < len);
        }
      }
    }
  }
  while (cumulative_number_of_bytes_sent < len);

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

  if (rtv == (-1))
  {
    return -1;
  }

  return (ssize_t)cumulative_number_of_bytes_sent;
}

inline ssize_t mws_recv(fd_t fd,
                        void* buf_ptr,
                        size_t len,
                        int flags)
{
  return recv(fd, buf_ptr, len, flags);
}

inline int mws_close(fd_t fd)
{
  return close(fd);
}
