#ifndef MWS_TYPE_DEFINITION_H_INCLUDED
#define MWS_TYPE_DEFINITION_H_INCLUDED

#include <ctime>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <map>
#include <pthread.h>
#include <sys/socket.h>
#include <string>
#include <netinet/tcp.h>
#include <vector>

#include "./mws_fast_deque.h"

// ctx 最小的 stack size.
#define MIN_MWS_CTX_STACK_SIZE 131072

// 傳送與接收緩衝區的大小(byte).
#define SND_RECV_BUFFER_SIZE 114688

// 一個訊息純資料部分 (不包括 head) 的大小為 1 KB.
#define MAX_MSG_SIZE 1024

// 呼叫 recv() 的時候一次可以讀進來的資料量 (硬體限制) 為 16 KB.
#define MAX_RECV_SIZE 16384

// fd list 可以儲存的 fd 最大值.
#define MAX_FD_SIZE 1024

// topic name 最大為 64 Byte.
#define MAX_TOPIC_NAME_SIZE 64

// second(s) for session timed out.
#define SESSION_TIMED_OUT_SEC 20

// receiver 做 topic check 最大秒數.
#define RCV_TOPIC_CHECK_TIMED_OUT_SEC 20

#define MAX_NUM_OF_RETRIES_SEND 10

#define MAX_NUM_OF_RETRIES_RECV_MSG 1
#define MAX_NUM_OF_RETRIES_RECV_TOPIC_NAME 10
#define MAX_NUM_OF_RETRIES_RECV_FFFD_FEFC 3

// struct sockaddr_in
// {
//     short            sin_family;   // e.g. AF_INET.
//     unsigned short   sin_port;     // e.g. htons(3490).
//     struct in_addr   sin_addr;     // see struct in_addr, below.
//     char             sin_zero[8];  // zero this if you want to.
// };
// struct in_addr
// {
//    uint32_t       s_addr;     // address in network byte order.
// };
typedef sockaddr_in sockaddr_in_t;
// struct sockaddr
// {
//   unsigned short  sa_family;  // 2 bytes address family, AF_xxx.
//   char       sa_data[14];     // 14 bytes of protocol address.
// };
typedef sockaddr sockaddr_t;

typedef timeval timeval_t;
typedef int fd_t;

class mws_evq;
typedef mws_evq mws_evq_t;

class mws_evq_attr;
typedef mws_evq_attr mws_evq_attr_t;

class mws_ctx;
typedef mws_ctx mws_ctx_t;

class mws_ctx_attr;
typedef mws_ctx_attr mws_ctx_attr_t;

class mws_reactor_only_ctx;
typedef mws_reactor_only_ctx mws_reactor_only_ctx_t;

class mws_reactor_only_ctx_attr;
typedef mws_reactor_only_ctx_attr mws_reactor_only_ctx_attr_t;

class mws_src;
typedef mws_src mws_src_t;

class mws_src_attr;
typedef mws_src_attr mws_src_attr_t;

class mws_rcv;
typedef mws_rcv mws_rcv_t;

class mws_rcv_attr;
typedef mws_rcv_attr mws_rcv_attr_t;

// 一組 IP:port.
struct ip_port
{
  std::string IP;
  uint16_t port;

  // Comparison operators/relational operators
  // equal to operator == overloading for ip_port
  bool operator ==(const ip_port& addr) const;
};
typedef ip_port ip_port_t;

// 一組 std::string 格式的 IP:port.
struct str_ip_port
{
  std::string str_ip;
  std::string str_port;
};
typedef str_ip_port str_ip_port_t;

// IP:low ~ high (用於 sess_addr_pair).
struct ip_port_low_high
{
  std::string IP;
  uint16_t low_port;
  uint16_t high_port;
  // next_bind_port: 從 next_bind_port 開始 bind.
  uint16_t next_bind_port;
};
typedef ip_port_low_high ip_port_low_high_t;

// listen IP:port + conn IP:port pair of a session.
struct sess_addr_pair
{
  ip_port_low_high_t listen_addr;
  ip_port_low_high_t conn_addr;

  // Constructor.
  sess_addr_pair();
  // basic assignment operator = overloading for sess_addr_pair_t
  void operator =(const sess_addr_pair& addr_pair);
};
typedef sess_addr_pair sess_addr_pair_t;

struct wait_to_connect_rcv_session
{
  mws_rcv_t* rcv_ptr;
  sess_addr_pair_t rcv_connection_setting;
  uint16_t next_port;
  uint64_t try_cnt;
};
typedef wait_to_connect_rcv_session wait_to_connect_rcv_session_t;

// rcv check topic 方法:
// 1. rcv 循環送 FF, 直到收到 src 的 FE 或 FC, 到步驟 2.
// 2. 送一個 FD 並等待 src 送來的 FC, 若收到 FC, 到步驟 3.
// 3. 送出 topic name 檢核訊息, 並等待 src 的 topic name 檢核訊息,
//    若收到 src 的 topic name 檢核訊息, 到步驟 4.
// 4. 檢核 topic name 是否正確:
//    topic name 檢核正確, 產生 MWS_MSG_BOS 事件, 檢核結束.
//    topic name 檢核錯誤, 斷線.
// src check topic 方法:
// 1. src 循環送 FE, 直到收到 src 的 FF 或 FD, 到步驟 2.
// 2. 送一個 FC 並等待 rcv 送來的 FD, 若收到 FD, 到步驟 3.
// 3. 送出 topic name 檢核訊息, 並等待 rcv 的 topic name 檢核訊息,
//    若收到 rcv 的 topic name 檢核訊息, 到步驟 4.
// 4. 檢核 topic name 是否正確:
//    topic name 檢核正確, 產生 MWS_SRC_EVENT_CONNECT 事件, 檢核結束.
//    topic name 檢核錯誤, 斷線.

struct wait_to_check_topic_src_conn_session
{
  fd_t fd;
  mws_src_t* src_ptr;
  sockaddr_in_t src_conn_listen_addr_info;
  //sockaddr_in_t src_conn_addr_info;
  sockaddr_in_t src_conn_rcv_addr_info;
};
typedef wait_to_check_topic_src_conn_session wait_to_check_topic_src_conn_session_t;

struct wait_to_check_topic_rcv_session
{
  fd_t fd;
  mws_rcv_t* rcv_ptr;
  sockaddr_in_t rcv_listen_addr_info;
  //sockaddr_in_t rcv_src_conn_addr_info;
  sockaddr_in_t rcv_addr_info;

  time_t t_starting_time;
};
typedef wait_to_check_topic_rcv_session wait_to_check_topic_rcv_session_t;

// 傳送訊息時只需標明是 msg 還是 hb, 在 dispatch 過程中會藉由收到訊息的 fd 的 role,
// 將 event_type 改成 MWS_SRC_DATA / MWS_MSG_DATA.
// (傳送方沒辦法幫接收方標明是哪一種 event_type).
// message event 類別.
#define MSG_TYPE_MSG     1
// heartbeat event 類別.
#define MSG_TYPE_HB      2
// 傳輸 message 的格式的 head (size: 16).
struct mws_pkg_head
{
  // sequence number of message.
  uint64_t seq_num;
  // length of message.
  uint16_t msg_size;
  // alignment issue.
  char filler_1[2];
  // type of message.
  uint8_t msg_type;
  // alignment issue.
  char filler_2[3];

  // Constructor.
  mws_pkg_head();
};
typedef mws_pkg_head mws_pkg_head_t;

// 傳輸 message 的格式.
// 注意: 傳送時 body 只傳送有資料的部分, 不會傳送整個 mws_pkg_t.
struct mws_pkg
{
  mws_pkg_head_t head;
  char body[MAX_MSG_SIZE];
};
typedef mws_pkg mws_pkg_t;

// 傳輸 topic name 檢查訊息的格式
// 注意: 傳送時完整傳送整個 mws_topic_name_msg_t.
struct mws_topic_name_msg
{
  // 要以 network (Big-endian) 形式存放, 以 host (平台決定) 方式讀出使用.
  uint16_t topic_name_len;
  char topic_name[MAX_TOPIC_NAME_SIZE];
};
typedef mws_topic_name_msg mws_topic_name_msg_t;

// Begin: event_type 定義.
// 以下為 source event (src conn fd) 類別:
#define MWS_SRC_DATA                      0
#define MWS_SRC_EVENT_CONNECT             1
#define MWS_SRC_EVENT_DISCONNECT          2
#define MWS_SRC_UNRECOVERABLE_LOSS        3
// 以下為 receiver event (rcv fd) 類別:
#define MWS_MSG_DATA                     50
#define MWS_MSG_BOS                      51
#define MWS_MSG_EOS                      52
#define MWS_MSG_UNRECOVERABLE_LOSS       53

// End: event_type 定義.
struct mws_event
{
  // event 所屬的 fd.
  fd_t fd;

  // event 類別.
  uint8_t event_type;

  // topic name.
  std::string topic_name;

  mws_src_t* src_ptr;
  mws_rcv_t* rcv_ptr;

  // src 用以 lieten 的 ip/port.
  str_ip_port_t src_addr;
  // rcv 的 ip/port.
  str_ip_port_t rcv_addr;

  // **********************************************************************
  // 關於 MWS_SRC_UNRECOVERABLE_LOSS 和 MWS_MSG_UNRECOVERABLE_LOSS 時,
  // seq_num 和 loss_num 的說明:
  // 當 上一筆資料 seq_num 為 n, 本筆資料 seq_num 為 n + x, 且 x > 1,
  // 則產生 MWS_SRC_UNRECOVERABLE_LOSS 和 MWS_MSG_UNRECOVERABLE_LOSS.
  // 此時 MWS_SRC_UNRECOVERABLE_LOSS 和 MWS_MSG_UNRECOVERABLE_LOSS event 中
  // seq_num 值為 n + 1, loss_num 值為 x - 1;
  // 代表自 seq_num 筆資料開始共掉了 loss_num 筆資料.
  // **********************************************************************

  // sequence number of message.
  uint64_t seq_num;
  // number of loss message.
  uint64_t loss_num;

  // message size.
  size_t msg_size;
  // message.
  char msg[MAX_MSG_SIZE];
};
typedef mws_event mws_event_t;

// mws_event_t*: event pointer, void*: cunstom data pointer, size_t cunstom data size.
typedef int (callback_t)(mws_event_t*, void*, size_t);

// src conn fd 的 receive buffer 預設配置 block 的數量.
#define SRC_CONN_NUM_OF_DEFAULT_RECV_BUFFER_BLOCK 100
// src conn fd 的 receive buffer 再次配置 block 的數量.
#define SRC_CONN_NUM_OF_EXT_RECV_BUFFER_BLOCK 100
// src conn fd 的 send buffer 預設配置 block 的數量.
#define SRC_CONN_NUM_OF_DEFAULT_SND_BUFFER_BLOCK 20
// src conn fd 的 send buffer 再次配置 block 的數量.
#define SRC_CONN_NUM_OF_EXT_SND_BUFFER_BLOCK 10

// rcv fd 的 receive buffer 預設配置 block 的數量.
#define RCV_NUM_OF_DEFAULT_RECV_BUFFER_BLOCK 10000
// rcv fd 的 receive buffer 再次配置 block 的數量.
#define RCV_NUM_OF_EXT_RECV_BUFFER_BLOCK 1000
// rcv fd 的 send buffer 預設配置 block 的數量.
#define RCV_NUM_OF_DEFAULT_SND_BUFFER_BLOCK 10
// rcv fd 的 send buffer 再次配置 block 的數量.
#define RCV_NUM_OF_EXT_SND_BUFFER_BLOCK 10

// mws_fd_detail 的 role 定義:
// FD_ROLE_UNKNOWN: 尚未使用.
#define FD_ROLE_UNKNOWN 0
// FD_ROLE_SRC_LISTEN: src listen fd.
#define FD_ROLE_SRC_LISTEN 1
// FD_ROLE_SRC_CONN: src connect fd.
#define FD_ROLE_SRC_CONN 2
// FD_ROLE_RCV: rcv fd.
#define FD_ROLE_RCV 3
// mws_fd_detail 的 status 定義:
// FD_ROLE_UNKNOWN 的 status: 此 fd 尚未使用.
#define FD_STATUS_UNKNOWN 0
// FD_ROLE_SRC_LISTEN 的 status: 可以 accept connection.
#define FD_STATUS_SRC_LISTEN_READY 11
// FD_ROLE_SRC_LISTEN 的 status: 準備 close, 不可再做其他動作.
#define FD_STATUS_SRC_LISTEN_WAIT_TO_CLOSE 19
// FD_ROLE_SRC_CONN 的 status: 等待切換為 FD_STATUS_SRC_CONN_WAIT_FFFD.
#define FD_STATUS_SRC_CONN_PREPARE 21
// FD_ROLE_SRC_CONN 的 status: check topic - 等待 FF/FD.
#define FD_STATUS_SRC_CONN_WAIT_FFFD 22
// FD_ROLE_SRC_CONN 的 status: check topic - 等待 FD.
#define FD_STATUS_SRC_CONN_WAIT_FD 23
// FD_ROLE_SRC_CONN 的 status: check topic - 等待 topic name.
#define FD_STATUS_SRC_CONN_WAIT_TOPIC_NAME 24
// FD_STATUS_SRC_CONN_TOPIC_CHECK_OK 的 status: 等待切換至 FD_STATUS_SRC_CONN_READY 之前不處理接收資料.
#define FD_STATUS_SRC_CONN_TOPIC_CHECK_OK 25
// FD_ROLE_SRC_CONN 的 status: 可以接收或傳送資料.
#define FD_STATUS_SRC_CONN_READY 26
// FD_ROLE_SRC_CONN 的 status: 該連線已經無法使用, 等待產生完所有 event 進入 FD_STATUS_SRC_CONN_WAIT_TO_CLOSE.
#define FD_STATUS_SRC_CONN_FD_FAIL 27
// FD_ROLE_SRC_CONN 的 status: 準備 close, 不可再做其他動作.
#define FD_STATUS_SRC_CONN_WAIT_TO_CLOSE 29
// FD_ROLE_RCV 的 status: 等待切換為 FD_STATUS_RCV_WAIT_FEFC.
#define FD_STATUS_RCV_PREPARE 31
// FD_ROLE_RCV 的 status: check topic - 等待 FE/FC.
#define FD_STATUS_RCV_WAIT_FEFC 32
// FD_ROLE_RCV 的 status: check topic - 等待 FC.
#define FD_STATUS_RCV_WAIT_FC 33
// FD_ROLE_RCV 的 status: check topic - 等待 topic name.
#define FD_STATUS_RCV_WAIT_TOPIC_NAME 34
// FD_STATUS_RCV_TOPIC_CHECK_OK 的 status: 等待切換至 FD_STATUS_RCV_READY 之前不處理接收資料.
#define FD_STATUS_RCV_TOPIC_CHECK_OK 35
// FD_ROLE_RCV 的 status: 可以接收或傳送資料.
#define FD_STATUS_RCV_READY 36
// FD_ROLE_RCV 的 status: 該連線已經無法使用, 等待產生完所有 event 進入 FD_STATUS_RCV_WAIT_TO_CLOSE.
#define FD_STATUS_RCV_FD_FAIL 37
// FD_ROLE_RCV 的 status: 準備 close, 不可再做其他動作.
#define FD_STATUS_RCV_WAIT_TO_CLOSE 39

struct mws_fd_detail
{
  pthread_mutex_t mutex;

  // file descriptor.
  fd_t fd;
  // fd 的角色:
  int16_t role;
  // fd 的狀態:
  int16_t status;

  // Begin: FD_ROLE_SRC_LISTEN 所使用的變數.
  // 指向 src 的指標.
  mws_src_t* src_listen_ptr;
  // src bind 的 listen address info.
  sockaddr_in_t src_listen_addr_info;
  // End: FD_ROLE_SRC_LISTEN 所使用的變數.

  // Begin: FD_ROLE_SRC_CONN 所使用的變數.
  // 指向 src 的指標.
  mws_src_t* src_conn_ptr;
  // src bind 的 listen address info.
  sockaddr_in_t src_conn_listen_addr_info;
  // src 給予 rcv 真實連線的 src conn address info.
  //sockaddr_in_t src_conn_addr_info;
  // 與 src conn fd 連線的 rcv address info.
  sockaddr_in_t src_conn_rcv_addr_info;
  // src 是否已經送過 0xFC 給 rcv.
  bool src_conn_sent_FC;
  // src 是否已經送過 topic name 給 rcv.
  bool src_conn_sent_topic_name;
  // End: FD_ROLE_SRC_CONN 所使用的變數.

  // Begin: FD_ROLE_RCV 所使用的變數.
  // 指向 rcv 的指標.
  mws_rcv_t* rcv_ptr;
  // src 的 listen address info.
  sockaddr_in_t rcv_listen_addr_info;
  // src 給予 rcv 真實連線的 src conn address info.
  //sockaddr_in_t rcv_src_conn_addr_info;
  // rcv 自己 bind 的 address info.
  sockaddr_in_t rcv_addr_info;
  // rcv 是否已經送過 0xFD 給 rcv.
  bool rcv_sent_FD;
  // rcv 是否已經送過 topic name 給 src.
  bool rcv_sent_topic_name;
  // rcv 斷線後要重新連線的設定.
  sess_addr_pair_t rcv_connection_setting;
  // End: FD_ROLE_RCV 所使用的變數.

  // 每一個 fd 有一個 receive buffer,
  // 每一個 element 放置一次 receive 到的資料以及上一個不完整訊息的空間.
  mws_fast_deque_t* msg_evq_ptr;
  ssize_t msg_evq_number;

  // 每一個 fd 有一個 send buffer,
  // 用以放置在 non-block 模式未送出的訊息.
  mws_fast_deque_t* send_buffer_ptr;
  ssize_t send_buffer_number;

  // 最後一個 heartbeat 時間 (time_t 格式).
  time_t last_heartbeat_time;

  // 功能: 初始化 mws_fd_detail_t 變數, mutex 除外.
  // 回傳值: 無.
  // 參數 option_del_deque:
  //   true: 解構 msg_evq_ptr 和 send_buffer_ptr 指向的物件.
  //   false: 不解構 msg_evq_ptr 和 send_buffer_ptr 指向的物件.
  void fd_init(bool option_del_deque);

  #if (MWS_DEBUG == 1)
    void fd_lock(const std::string file, const std::string function, const int line_no);
    int fd_trylock(const std::string file, const std::string function, const int line_no);
    void fd_unlock(const std::string file, const std::string function, const int line_no);
  #else
    void fd_lock();
    int fd_trylock();
    void fd_unlock();
  #endif

  // 功能: 取得該 fd 的最後一個 heartbeat 時間 (time_t 格式).
  // 回傳值: 該 fd 的最後一個 heartbeat 時間 (time_t 格式).
  // 參數: 沒有.
  time_t get_last_heartbeat_time();
};
typedef mws_fd_detail mws_fd_detail_t;

struct mws_evq_id
{
  int evq_no;
  mws_evq_t* evq_ptr;
};
typedef mws_evq_id mws_evq_id_t;

struct mws_msg_evq_buffer
{
  uint32_t filler_1;
  // buffer 的資料的起始點.
  size_t begin_pos;
  uint32_t filler_2;
  // buffer 資料的長度.
  size_t data_size;
  uint32_t filler_3;
  // sizeof(mws_pkg_t) + MAX_RECV_SIZE.
  char buffer[sizeof(mws_pkg_t) + MAX_RECV_SIZE];
};
typedef mws_msg_evq_buffer mws_msg_evq_buffer_t;

// src/rcv 傳送資料以 FLUSH (BLOCK) 模式傳送.
#define MWS_SEND_MSG_FLUSH  0b00000000000000000000000000000000

// 儲存未送出 message 的 structure 格式.
struct mws_send_buff
{
  ssize_t pos;
  ssize_t remaining_msg_len;
  char buff[sizeof(mws_pkg_t)];
};
typedef mws_send_buff mws_send_buff_t;

// No error.
#define MWS_NO_ERROR                        0
// Create pthread error.
#define MWS_ERROR_PTHREAD_CREATE            1
// Initialize pthread mutex error.
#define MWS_ERROR_PTHREAD_MUTEX_INIT        2
// Topic name error.
#define MWS_ERROR_TOPIC_NAME                3
// Create listen socket error.
#define MWS_ERROR_LISTEN_SOCKET_CREATE      4
// Create connect socket error.
#define MWS_ERROR_CONNECT_SOCKET_CREATE     5

#endif // MWS_TYPE_DEFINITION_H_INCLUDED
