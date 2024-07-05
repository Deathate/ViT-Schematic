#ifndef MWS_UTIL_H_
#define MWS_UTIL_H_

// 功能: 建立 listen socket.
// 回傳值: 0: 建立 listen socket 成功.
//         -1: 建立 listen socket 失敗.
//         -2: 設定 TCP_NODELAY 失敗.
//         -3: 設定 send buffer size 失敗.
//         -4: 設定 receive buffer size 失敗.
//         -5: bind listen address 失敗.
//         -6: listen 失敗.
// 參數: src_ptr: 要建立的 listen socket 所屬於的 src.
int create_listen_socket(mws_src_t* src_ptr);

// 功能: 建立 connect socket (包含 socket(), bind() connect()).
// 回傳值 0: 建立 connect socket 成功.
// 回傳值 -1: 建立 connect socket 失敗.
// 回傳值 -2: 設定 TCP_NODELAY 失敗.
// 回傳值 -3: 設定 send buffer size 失敗.
// 回傳值 -4: 設定 receive buffer size 失敗.
// 回傳值 -5: bind connect address 失敗.
// 回傳值 -6: requect connection 失敗.
// 參數 &sess_info: 包含要建立的 connect socket 所屬於的 rcv 和要連線的 session 資訊.
// 參數 &rcv_listen_addr_info: 連線的相對方 src 的 address info.
// 參數 &rcv_conn_addr_info: 連線的 rcv 本身 bind 的 address info.
int create_connect_socket(wait_to_connect_rcv_session_t& sess_info,
                          sockaddr_in_t& rcv_listen_addr_info,
                          sockaddr_in_t& rcv_conn_addr_info);

// 功能: src 接受 rcv 的連線要求.
// 回傳值: 無.
// 參數: ctx_ptr: 當前的 ctx.
//       selected_fd: 要接受連線要求的 fd 值.
void step_accept_connection(mws_ctx_t* ctx_ptr, fd_t selected_fd);

// 功能: rcv 向 src 要求連線.
// 回傳值 0: 連線成功.
// 回傳值 -1: 連線失敗.
// 回傳值 -2: 在 TANDEM 環境下 connect 回傳成功但實際上 socket 已經發生錯誤, 需要重建 socket.
// 參數 ctx_ptr: 當前的 ctx.
// 參數 conn_fd: 要求連線的 fd 值.
// 注意: 只能從 create_connect_socket() 呼叫.
int request_connection(mws_ctx_t* ctx_ptr, fd_t conn_fd);

// 功能: 設定 dest 的 port range.
//       port low/high 必須在 1 - 65535 區間.
//       如果 high < low 則自動對調並刷 mws log.
//       如果對調後 low < 1 則設為 1 並刷 mws log.
//       如果對調後 high > 65535 則設為 65535 並刷 mws log.
// 回傳值: 無.
// 參數 dest: 要設定 port range 的物件.
// 參數 port_low: port range 最小值.
// 參數 port_high: port range 最大值.
// 參數 source_file_of_caller: 在呼叫的地方填入 __FILE__.
// 參數 function_of_caller: 在呼叫的地方填入 __func__.
// 參數 line_no_of_caller: 在呼叫的地方填入 __LINE__.
void set_port_high_low(ip_port_low_high_t& dest,
                       uint16_t port_low,
                       uint16_t port_high,
                       const std::string source_file_of_caller,
                       const std::string function_of_caller,
                       const int line_no_of_caller);

// 功能: 將 sockaddr_in_t 內的 IP/port 分別轉為 std::string.
// 回傳值: 無.
// 參數 addr_info: sockaddr_in_t 格式的來源變數.
// 參數 &str_ip: std::string 格式的 IP.
// 參數 &str_port: std::string 格式的 port.
void sockaddr_in_t_to_string(const sockaddr_in_t addr_info,
                             std::string& str_ip,
                             std::string& str_port);

// 功能: 讀取傳送到屬於參數 fd 的 socket 內的資料到 recv_buff_ptr 指向的空間.
// 回傳值 > 0: 讀取到的 byte 數.
// 回傳值 0: 對方斷線.
// 回傳值 -1: socket 出現錯誤, 要檢查 errno 了解詳細原因.
// 回傳值 -2: 連續發生 EAGAIN/EWOULDBLOCK/EINTR 超過 max_retry_cnt 次.
// 參數 *recv_buff_ptr: 指向讀取到的資料要放置的空間的指標.
// 參數 fd: 從此 fd 對應的 socket 讀取資料.
// 參數 len: 讀取資料的最大長度.
// 參數 max_retry_cnt: 當發生 EAGAIN/EWOULDBLOCK/EINTR 時, 最多嘗試 recv 幾次.
ssize_t recv_data(void* recv_buff_ptr,
                  const fd_t fd,
                  const size_t max_len,
                  int32_t max_retry_cnt);

// 功能: 處理 rcv 收到 FE 或 FC.
// 回傳值: 無.
// 參數 &it: rcv fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_rcv_wait_fefc(std::deque<fd_t>::iterator& it);

// 功能: 處理 rcv 收到 topic name.
// 回傳值: 無.
// 參數 &it: rcv fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_rcv_wait_topic_name(std::deque<fd_t>::iterator& it);

// 功能: 處理 rcv 收到 message.
// 回傳值: 無.
// 參數 &it: rcv fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_rcv_ready(std::deque<fd_t>::iterator& it);

// 功能: 處理 src conn 收到 FF 或 FD.
// 回傳值: 無.
// 參數 &it: src conn fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_src_conn_wait_fffd(std::deque<fd_t>::iterator& it);

// 功能: 處理 src conn 收到 topic name.
// 回傳值: 無.
// 參數 &it: src conn fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_src_conn_wait_topic_name(std::deque<fd_t>::iterator& it);

// 功能: 處理 src conn 收到 message.
// 回傳值: 無.
// 參數 &it: src conn fds 的 iterator.
// 注意: 使用完此函數需自行檢查 iterator 是否為 end.
void step_src_conn_ready(std::deque<fd_t>::iterator& it);

// 功能: 更新 g_fd_table 某個 fd 資料的 status.
// 回傳值: 無.
// 參數 fd: 要更新 g_fd_tablefd 中屬於哪個 fd 的資料.
// 參數 new_statu: 新的 status.
// 參數 function: 呼叫方的函式名稱 (可以 __func__ 帶入).
// 參數 line_no: 呼叫方的行號 (可以 __LINE__ 帶入).
void update_g_fd_table_status(const fd_t fd,
                              const int16_t new_status,
                              const std::string function,
                              const int line_no);

// 功能: 傳送 send_buff_ptr 指向的空間內的資料到屬於參數 fd 的 socket.
// 回傳值 >= 0: 傳送的 byte 數.
// 回傳值 < 0: socket 出現錯誤, 要檢查 errno 了解詳細原因.
// 參數 *send_buff_ptr: 指向要傳送的資料放置的空間的指標.
// 參數 fd: 從此 fd 對應的 socket 傳送資料.
// 參數 len: 傳送資料的最大長度.
ssize_t send_topic_check_code(void* send_buff_ptr,
                              const fd_t fd,
                              const size_t max_len);

// 功能: 產生參數 fd 對應的 socket 的詳細資料.
// 回傳值: 無.
// 參數 fd: 從此 fd 對應的 socket 產生 log_body.
// 參數 log_body: 此 fd 對應的 socket 的詳細資料.
void fd_info_log(const fd_t fd,
                 std::string& log_body);

// 功能: src conn 傳送 FE 發生錯誤.
// 回傳值: 無.
// 參數 &it: wait_to_check_topic_src_conn_session_t 的 iterator.
// 參數 function: 發生錯誤的函式.
// 參數 line_no: 發生錯誤的行號.
void step_send_fe_error(std::deque<wait_to_check_topic_src_conn_session_t>::iterator& it,
                        const std::string function,
                        const int line_no);

// 功能: rcv 傳送 FF 發生錯誤.
// 回傳值: 無.
// 參數 &it: wait_to_check_topic_rcv_session_t 的 iterator.
// 參數 function: 發生錯誤的函式.
// 參數 line_no: 發生錯誤的行號.
void step_send_ff_error(std::deque<wait_to_check_topic_rcv_session_t>::iterator& it,
                        const std::string function,
                        const int line_no);

// 功能: 依設定建立 rcv socket 並 connect src.
// 回傳值: 無.
// 參數 &it: wait_to_connect_rcv_session_t 的 iterator.
// 參數 *ctx_ptr: 指向所使用的 ctx 的指標.
// 參數 function: 發生錯誤的函式.
// 參數 line_no: 發生錯誤的行號.
void step_rcv_connect(std::deque<wait_to_connect_rcv_session_t>::iterator& it,
                      mws_ctx_t* ctx_ptr,
                      const std::string function,
                      const int line_no);

void rcv_topic_check_timeout_error(fd_t fd,
                                   const std::string function,
                                   const int line_no);

#endif /* MWS_UTIL_H_ */
