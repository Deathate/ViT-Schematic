#ifndef MWS_DEBUG_H_
#define MWS_DEBUG_H_

#include <string>

#include "../inc/mws_type_definition.h"

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
               bool show_all_set_and_max_fd);

void* mws_debug_fun(void* data);

#endif /* MWS_DEBUG_H_ */
