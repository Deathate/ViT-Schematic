//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_DEBUG_CPP 1

#include <deque>
#include <iostream>
#include <sys/socket.h>
#include <string>
#include <unistd.h>

#include "../inc/mws_class_definition.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_type_definition.h"

using namespace mws_global_variable;

void ctx_debug(mws_ctx_t* ctx_ptr,
               const std::string function,
               const int line_no,
               bool show_ctx_list_wait_to_connect_rcv_session,
               bool show_ctx_list_wait_to_check_topic_src_conn_session,
               bool show_ctx_list_wait_to_check_topic_rcv_session,
               bool show_ctx_list_wait_to_close_src_conn_fds,
               bool show_ctx_list_wait_to_close_rcv_fds,
               bool show_ctx_list_owned_src_listen_fds,
               bool show_ctx_list_owned_src_conn_fds,
               bool show_ctx_list_owned_rcv_fds,
               bool show_all_set_and_max_fd)
{
  std::cout << "***** " << function << " ln:" << std::to_string(line_no) << std::endl;

  // Begin: ctx_list_wait_to_connect_rcv_session.
  if (show_ctx_list_wait_to_connect_rcv_session == true)
  {
    std::cout << "*****¡@Begin: ctx_list_wait_to_connect_rcv_session" << std::endl;
    std::cout << "ctx_list_wait_to_connect_rcv_session.size():"
              << std::to_string( ctx_ptr->ctx_list_wait_to_connect_rcv_session.size() )
              << std::endl;

    std::cout << "ctx_list_wait_to_connect_rcv_session:" << std::endl;
    for (std::deque<wait_to_connect_rcv_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_connect_rcv_session.begin();
         it != ctx_ptr->ctx_list_wait_to_connect_rcv_session.end();
         ++it)
    {
      std::cout << it->rcv_ptr->topic_name
                << "/"
                << it->rcv_connection_setting.listen_addr.IP
                << ":"
                << std::to_string( it->rcv_connection_setting.listen_addr.low_port )
                << "-"
                << std::to_string( it->rcv_connection_setting.listen_addr.high_port )
                << "/"
                << it->rcv_connection_setting.conn_addr.IP
                << ":"
                << std::to_string( it->rcv_connection_setting.conn_addr.low_port )
                << "-"
                << std::to_string( it->rcv_connection_setting.conn_addr.high_port )
                << std::endl;
    }

    std::cout << "*****¡@End: ctx_list_wait_to_connect_rcv_session" << std::endl;
  }
  // End: ctx_list_wait_to_connect_rcv_session.

  // Begin: ctx_list_wait_to_check_topic_src_conn_session.
  if (show_ctx_list_wait_to_check_topic_src_conn_session == true)
  {
    std::cout << "*****¡@Begin: ctx_list_wait_to_check_topic_src_conn_session" << std::endl;
    std::cout << "ctx_list_wait_to_check_topic_src_conn_session.size():"
              << std::to_string( ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.size() )
              << std::endl;
    std::cout << "ctx_list_wait_to_check_topic_src_conn_session fds:";
    for (std::deque<wait_to_check_topic_src_conn_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.begin();
         it != ctx_ptr->ctx_list_wait_to_check_topic_src_conn_session.end();
         ++it)
    {
      std::cout << std::to_string( it->fd ) << " ";
    }
    std::cout << std::endl;
    std::cout << "*****¡@End: ctx_list_wait_to_check_topic_src_conn_session" << std::endl;
  }
  // End: ctx_list_wait_to_check_topic_src_conn_session.

  // Begin: ctx_list_wait_to_check_topic_rcv_session.
  if (show_ctx_list_wait_to_check_topic_rcv_session == true)
  {
    std::cout << "*****¡@Begin: ctx_list_wait_to_check_topic_rcv_session" << std::endl;
    std::cout << "ctx_list_wait_to_check_topic_rcv_session.size():"
              << std::to_string( ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.size() )
              << std::endl;
    std::cout << "ctx_list_wait_to_check_topic_rcv_session fds:";
    for (std::deque<wait_to_check_topic_rcv_session_t>::iterator it = ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.begin();
         it != ctx_ptr->ctx_list_wait_to_check_topic_rcv_session.end();
         ++it)
    {
      std::cout << std::to_string( it->fd ) << " ";
    }
    std::cout << std::endl;
    std::cout << "*****¡@End: ctx_list_wait_to_check_topic_rcv_session" << std::endl;
  }
  // End: ctx_list_wait_to_check_topic_rcv_session.

  // Begin: ctx_list_wait_to_close_src_conn_fds.
  if (show_ctx_list_wait_to_close_src_conn_fds == true)
  {
    std::cout << "*****¡@Begin: ctx_list_wait_to_close_src_conn_fds" << std::endl;
    std::cout << "ctx_list_wait_to_close_src_conn_fds.size():"
              << std::to_string( ctx_ptr->ctx_list_wait_to_close_src_conn_fds.size() )
              << std::endl;
    std::cout << "ctx_list_wait_to_close_src_conn_fds fds:";
    for (std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_wait_to_close_src_conn_fds.begin();
         it != ctx_ptr->ctx_list_wait_to_close_src_conn_fds.end();
         ++it)
    {
      std::cout << std::to_string( *it ) << " ";
    }
    std::cout << std::endl;
    std::cout << "*****¡@End: ctx_list_wait_to_close_src_conn_fds" << std::endl;
  }
  // End: ctx_list_wait_to_close_src_conn_fds.

  // Begin: ctx_list_wait_to_close_rcv_fds.
  if (show_ctx_list_wait_to_close_rcv_fds == true)
  {
    std::cout << "*****¡@Begin: ctx_list_wait_to_close_rcv_fds" << std::endl;
    std::cout << "ctx_list_wait_to_close_rcv_fds.size():"
              << std::to_string( ctx_ptr->ctx_list_wait_to_close_rcv_fds.size() )
              << std::endl;
    std::cout << "ctx_list_wait_to_close_rcv_fds fds:";
    for (std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_wait_to_close_rcv_fds.begin();
         it != ctx_ptr->ctx_list_wait_to_close_rcv_fds.end();
         ++it)
    {
      std::cout << std::to_string( *it ) << " ";
    }
    std::cout << std::endl;
    std::cout << "*****¡@End: ctx_list_wait_to_close_rcv_fds" << std::endl;
  }
  // End: ctx_list_wait_to_close_rcv_fds.

  // Begin: ctx_list_owned_src_listen_fds.
  if (show_ctx_list_owned_src_listen_fds == true)
  {
    std::cout << "*****¡@Begin: ctx_list_owned_src_listen_fds" << std::endl;
    std::cout << "ctx_list_owned_src_listen_fds.size():"
              << std::to_string( ctx_ptr->ctx_list_owned_src_listen_fds.size() )
              << std::endl;
    std::cout << "ctx_list_owned_src_listen_fds fds:" << std::endl;
    for (std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_src_listen_fds.begin();
         it != ctx_ptr->ctx_list_owned_src_listen_fds.end();
         ++it)
    {
      std::cout << "fd: " << std::to_string( *it ) << ", status: " << g_fd_table[*it].status << std::endl;
    }
    std::cout << "*****¡@End: ctx_list_owned_src_listen_fds" << std::endl;
  }
  // End: ctx_list_owned_src_listen_fds.

  // Begin: ctx_list_owned_src_conn_fds.
  if (show_ctx_list_owned_src_conn_fds == true)
  {
    std::cout << "*****¡@Begin: ctx_list_owned_src_conn_fds" << std::endl;
    std::cout << "ctx_list_owned_src_conn_fds.size():"
              << std::to_string( ctx_ptr->ctx_list_owned_src_conn_fds.size() )
              << std::endl;
    std::cout << "ctx_list_owned_src_conn_fds fds:" << std::endl;
    for (std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_src_conn_fds.begin();
         it != ctx_ptr->ctx_list_owned_src_conn_fds.end();
         ++it)
    {
      std::cout << "fd: " << std::to_string( *it ) << ", status: " << g_fd_table[*it].status << std::endl;
    }
    std::cout << "*****¡@End: ctx_list_owned_src_conn_fds" << std::endl;
  }
  // End: ctx_list_owned_src_conn_fds.

  // Begin: ctx_list_owned_rcv_fds.
  if (show_ctx_list_owned_rcv_fds == true)
  {
    std::cout << "*****¡@Begin: ctx_list_owned_rcv_fds" << std::endl;
    std::cout << "ctx_list_owned_rcv_fds.size():"
              << std::to_string( ctx_ptr->ctx_list_owned_rcv_fds.size() )
              << std::endl;
    std::cout << "ctx_list_owned_rcv_fds fds:" << std::endl;
    for (std::deque<fd_t>::iterator it = ctx_ptr->ctx_list_owned_rcv_fds.begin();
         it != ctx_ptr->ctx_list_owned_rcv_fds.end();
         ++it)
    {
      std::cout << "fd: " << std::to_string( *it ) << ", status: " << g_fd_table[*it].status << std::endl;
    }
    std::cout << "*****¡@End: ctx_list_owned_rcv_fds" << std::endl;
  }
  // End: ctx_list_owned_rcv_fds.

  // Begin: all_set and max_fd.
  if (show_all_set_and_max_fd == true)
  {
    std::cout << "*****¡@Begin: all_set and max_fd" << std::endl;
    std::cout << "max_fd:"
              << std::to_string( ctx_ptr->max_fd )
              << std::endl;
    std::cout << "all_set fds:";
    for (fd_t fd = 0; fd <= ctx_ptr->max_fd; ++fd)
    {
      if (FD_ISSET(fd, &ctx_ptr->all_set))
      {
        std::cout << " " << std::to_string(fd);
      }
    }
    std::cout << std::endl;
    std::cout << "*****¡@End: all_set and max_fd" << std::endl;
  }
  // End: all_set and max_fd.

  return;
}

#if (MWS_DEBUG == 1)
  void* mws_debug_fun(void* data)
  {
    while (1)
    {
      sleep(300);
      std::deque<std::string>::iterator it;
      for (std::deque<std::string>::iterator it = g_mws_debug_log.begin();
           it != g_mws_debug_log.end();
           ++it)
      {
        std::cout << *it << "\n";
      }
      std::cout << std::endl;
      g_mws_debug_log.clear();

      {
        time_t curr_time = time(NULL);
        std::string log;
        log += __FILE__;
        log += " ";
        log += __func__;
        log += " ";
        log += std::to_string(__LINE__);
        log += " debug thread is alive: ";
        log += std::to_string(curr_time);
        std::cout << log << std::endl;
      }
    }
  
    //pthread_exit(NULL);
  }
#endif
