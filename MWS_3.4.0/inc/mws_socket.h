#ifndef MWS_SOCKET_H_
#define MWS_SOCKET_H_

#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include "./mws_type_definition.h"

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

ssize_t mws_send_nonblock(fd_t fd,
                          char* buf_ptr,
                          size_t len,
                          int flags);

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

#endif /* MWS_SOCKET_H_ */
