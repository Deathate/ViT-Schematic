// 日期      版本       維護人員    修改原因.
// 20240130  v01.00.00  吳青華      新程式開發.
//
// 系統名稱: 測試子系統.
// 程式代號: mwsrcv.
// 程式名稱: receiver sample code.
// 安全管制: 否.
// 開發語言: C++.
// 範例程式: 無.
//
// 程式功能: mwsrcv sample code.
//
// 執行時機: 沒有限制.
// RERUN   : 可.
//
// 程式介面:
//
// 介面種類        介 面 名 稱              中    文    說    明.
// ========  ===========================  =================================
// ARGV[1]   MWS config file path         MWS 設定檔位置.
// ARGV[2]   rcv cfg's section namee      rcv 設定在設定檔中的 section.
// ARGV[>3]  以下為選擇性參數:
//           -l [MWS log file path]       MWS log 檔位置.
//           -c [ctx's section_name]      ctx 設定在設定檔中的 section.
//           -e [evq's section_name]      evq 設定在設定檔中的 section.
//           -m [message(s) per second]   每秒送出幾筆 message(0-10000).
//           -s [size of a message]       每筆 message 的長度(1-1000).
//           -v                           顯示收到的每筆 message 的 HEX 內容.
//           -t                           顯示收到的每筆 message 的 text 內容.
//

#include <getopt.h>
#include <iostream>
#include <sstream>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <time.h>
#include <unistd.h>

#include "../../../inc/mws.h"
#include "../../lib/inc/Flow_Control.h"

using namespace std;

// MWS 相關變數.
// MWS context.
mws_ctx_t* g_ctx_ptr = NULL;
// MWS event queue.
mws_evq_t* g_evq_ptr = NULL;
// MWS receiver.
mws_rcv_t* g_rcv_ptr = NULL;

pthread_t g_mws_event_dispatch_thread;

mws_endianness_t g_endianness_obj;

// MWS 設定檔位置.
string g_mws_cfg_file_path("");
// rcv 設定在設定檔中的 section name.
string g_rcv_section_name("");
// MWS log 檔位置.
string g_mws_log_file_path("");
// ctx 設定在設定檔中的 section name.
string g_ctx_section_name("");
// evq 設定在設定檔中的 section name.
string g_evq_section_name("");
// 每秒送出幾筆 message.
int32_t g_msgs_per_second = 0;
// 每筆 message 的長度.
int16_t g_size_of_a_msg = 1;
// 是否顯示收到的每筆 message 的 HEX 內容.
bool g_show_msg_hex = false;
// 是否顯示收到的每筆 message 的 text 內容.
bool g_show_msg_text = false;

uint64_t g_received_bytes_during_this_sec = 0;
int16_t g_connected_cnt = 0;
uint64_t g_show_cnt = 0;
uint64_t g_seq_num_of_last_received_msg = 0;
pthread_mutex_t g_mutex;

void create_MWS_object(void);
//void close_MWS_object(void);

int rcv_callback(mws_event_t* event_ptr,
                 void* custom_data_ptr,
                 size_t custom_data_length);

void* mws_event_dispatch_thread_routine(void* client_data);

void handle_argument(int argc, char** argv);
bool check_argument();
void show_usage();

void debug_show_argument();

// 功能: 主函式, 程式進入點.
// 回傳值 0: 表示程式正常結束.
// 回傳值非 0: 表示程式不正常中斷.
// 參數 argc: 引數個數.
// 參數 argv: 引數陣列.
int main(int argc, char** argv)
{
  if (pthread_mutex_init(&g_mutex, NULL) == (-1))
  {
    string log_body = "pthread_mutex_init() 失敗";
    cerr << log_body << endl;
    exit(1);
  }

  handle_argument(argc, argv);

  //debug_show_argument();

  create_MWS_object();

  // Begin: create dispatch thread.
  {
    int rtv = pthread_create(&g_mws_event_dispatch_thread,
                             NULL,
                             mws_event_dispatch_thread_routine,
                             NULL);
    if (rtv != 0)
    {
      string log_body = "Dispatch thread 建立失敗";
      cerr << log_body << endl;
      exit(1);
    }
  }
  // End: create dispatch thread.

  if (g_msgs_per_second > 0)
  {
    // 設定 slice 大小.
    uint64_t time_slice_microsecond = 100000;
    if (g_msgs_per_second >= 1000)
    {
      time_slice_microsecond = 5000;
    }
    else if (g_msgs_per_second >= 20)
    {
      time_slice_microsecond = 50000;
    }
    // 設定流量控管.
    Flow_Control fc_obj((uint64_t)g_msgs_per_second * (uint64_t)g_size_of_a_msg * 10,
                        time_slice_microsecond,
                        1);

    cout << "Max fitting message size:" << fc_obj.show_max_fitting_message_size() << endl;;

    time_t t1 = time(NULL);
    time_t t2 = t1;
    int32_t number_of_msgs_sent_during_this_sec = 0;
    bool is_ready_to_send = false;
    while (1)
    {
      useconds_t required_delay_time = 1000;
      static uint64_t msg_number = 0;
      char msg[1000];

      if (number_of_msgs_sent_during_this_sec < g_msgs_per_second)
      {
        is_ready_to_send =
          fc_obj.flow_rate_test_and_consume_remaining_capacity((uint64_t)g_size_of_a_msg, required_delay_time);
      }
      else
      {
        is_ready_to_send = false;
      }

      pthread_mutex_lock(&g_mutex);
      if ((g_connected_cnt > 0) &&
          (is_ready_to_send == true))
      {
        pthread_mutex_unlock(&g_mutex);
        string msg_str = g_rcv_ptr->get_topic_name() + ":" + to_string(++msg_number);
        if ((int16_t)msg_str.length() > g_size_of_a_msg)
        {
          msg_str = msg_str.substr((msg_str.length() - g_size_of_a_msg), g_size_of_a_msg);
        }

        memset(&msg[0], '\0', 1000);
        memcpy(&msg[0],
               msg_str.c_str(),
               msg_str.length());

        int send_rtv = g_rcv_ptr->mws_hf_rcv_send((char*)&msg[0], g_size_of_a_msg, msg_number);
        if (send_rtv < 0)
        {
          printf("[SEND FAIL] errno: %d(%s)\n", errno, strerror(errno));
          pthread_mutex_unlock(&g_mutex);
          exit(1);
        }
        ++number_of_msgs_sent_during_this_sec;
      }
      else
      {
        pthread_mutex_unlock(&g_mutex);
        usleep(required_delay_time);
      }

      t2 = time(NULL);
      if (t1 != t2)
      {
        pthread_mutex_lock(&g_mutex);
        cout << "Received(" << ++g_show_cnt << "/"
             << g_seq_num_of_last_received_msg << "): "
             << g_received_bytes_during_this_sec << " bytes"<< endl;
        g_received_bytes_during_this_sec = 0;
        pthread_mutex_unlock(&g_mutex);

        number_of_msgs_sent_during_this_sec = 0;

        t1 = t2;
      }
    }
  }
  else
  {
    time_t t1 = time(NULL);
    time_t t2 = t1;

    while (1)
    {
      sleep(1);

      t2 = time(NULL);
      if (t1 != t2)
      {
        pthread_mutex_lock(&g_mutex);
        cout << "Received(" << ++g_show_cnt << "/"
             << g_seq_num_of_last_received_msg << "): "
             << g_received_bytes_during_this_sec << " bytes"<< endl;
        g_received_bytes_during_this_sec = 0;
        pthread_mutex_unlock(&g_mutex);
        t1 = t2;
      }
    }
  }

  // statement is unreachable
  //return 0;
}

// 功能: 1. 讀取並設定 MWS config file
//       2. 設定 MWS log file 位置
//       3. create MWS context
//       4. create MWS event queue
//       5. create MWS receiver
// 回傳值為 void.
// 沒有參數.
void create_MWS_object(void)
{
  // Begin: mws 初始化.
  int rtv = mws_init(string("mwsrcv"),
                     g_mws_cfg_file_path,
                     g_mws_log_file_path,
                     1);
  if (rtv == (-1))
  {
    string log_body = "mws 初始化失敗";
    cerr << log_body << endl;
    exit(1);
  }
  // End: mws 初始化.

  uint32_t returned_value = 0;

  // Begin: 建立 ctx.
  g_ctx_ptr = new mws_ctx_t(g_ctx_section_name);
  returned_value = g_ctx_ptr->mws_get_object_status();
  if (returned_value != MWS_NO_ERROR)
  {
    string log_body = "create mws_ctx 失敗";
    stringstream log_body_ss("");
    log_body_ss << "error: " << returned_value;
    log_body += log_body_ss.str();
    cerr << log_body << endl;
    exit(1);
  }
  // End: 建立 ctx.

  // Begin: 建立 evq.
  g_evq_ptr = new mws_evq_t(g_evq_section_name);
  returned_value = g_evq_ptr->mws_get_object_status();
  if (returned_value != MWS_NO_ERROR)
  {
    string log_body = "create mws_evq 失敗";
    stringstream log_body_ss("");
    log_body_ss << "error: " << returned_value;
    log_body += log_body_ss.str();
    cerr << log_body << endl;
    exit(1);
  }
  // End: 建立 evq.

  // Begin: 建立 rcv.
  g_rcv_ptr = new mws_rcv_t(g_rcv_section_name,
                            g_ctx_ptr,
                            g_evq_ptr,
                            rcv_callback);
  returned_value = g_rcv_ptr->mws_get_object_status();
  if (returned_value != MWS_NO_ERROR)
  {
    string log_body = "create mws_rcv 失敗";
    stringstream log_body_ss("");
    log_body_ss << "error: " << returned_value;
    log_body += log_body_ss.str();
    cerr << log_body << endl;
    exit(1);
  }
  // End: 建立 rcv.

  return ;
}

// 功能: 處理 receiver 的 event (內容由 AP 的程式設計師依照業務撰寫).
// 回傳值 0: 表示成功.
// 回傳值 非0: 表示失敗, 會跳離 dispatch_events() 和 mws_evq::mws_event_dispatch().
// 參數 event_ptr: 指向放置 event 的 buffer 的指標.
// 參數 custom_data_ptr: 指向放置 custom data 的 buffer 的指標.
// 參數 custom_data_length: custom data 的長度.
int rcv_callback(mws_event_t* event_ptr,
                 void* custom_data_ptr,
                 size_t custom_data_length)
{
  switch (event_ptr->event_type)
  {
    case MWS_MSG_DATA:
    {
      pthread_mutex_lock(&g_mutex);
      g_received_bytes_during_this_sec += event_ptr->msg_size;
      g_seq_num_of_last_received_msg = event_ptr->seq_num;
      pthread_mutex_unlock(&g_mutex);

      if (g_show_msg_hex == true)
      {
        cout << "HEX: ";
        for (size_t i = 0; i < event_ptr->msg_size; ++i)
        {
          printf("%02X ", (uint16_t)event_ptr->msg[i]);
        }
        cout << endl;
      }
      if (g_show_msg_text == true)
      {
        //string str(event_ptr->msg, event_ptr->msg_size);
        string str(event_ptr->msg);
        cout << "TEXT: " << str << endl;
      }

      break;
    }
    case MWS_MSG_BOS:
    {
      pthread_mutex_lock(&g_mutex);
      ++g_connected_cnt;
      pthread_mutex_unlock(&g_mutex);

      printf("[MWS_MSG_BOS]===============\n");
      cout << "fd: " << event_ptr->fd << "\n";
      cout << "topic_name: " << event_ptr->topic_name << "\n";
      cout << "src_addr ( " << event_ptr->src_addr.str_ip << ": " << event_ptr->src_addr.str_port << " )\n";
      cout << "rcv_addr ( " << event_ptr->rcv_addr.str_ip << ": " << event_ptr->rcv_addr.str_port << " )" << endl;
      printf("----------------------------\n");

      break;
    }
    case MWS_MSG_EOS:
    {
      pthread_mutex_lock(&g_mutex);
      --g_connected_cnt;
      cout << "Received(" << ++g_show_cnt << "): " << g_received_bytes_during_this_sec << " bytes"<< endl;
      g_received_bytes_during_this_sec = 0;
      pthread_mutex_unlock(&g_mutex);

      printf("[MWS_MSG_EOS]===============\n");
      cout << "fd: " << event_ptr->fd << "\n";
      cout << "topic_name: " << event_ptr->topic_name << "\n";
      cout << "src_addr ( " << event_ptr->src_addr.str_ip << ": " << event_ptr->src_addr.str_port << " )\n";
      cout << "rcv_addr ( " << event_ptr->rcv_addr.str_ip << ": " << event_ptr->rcv_addr.str_port << " )" << endl;
      printf("----------------------------\n");

      break;
    }
    case MWS_MSG_UNRECOVERABLE_LOSS:
    {
      printf("[MWS_MSG_UNRECOVERABLE_LOSS]====\n");
      cout << "fd: " << event_ptr->fd << "\n";
      cout << "topic_name: " << event_ptr->topic_name << "\n";
      cout << "rcv_addr ( " << event_ptr->rcv_addr.str_ip << ": " << event_ptr->rcv_addr.str_port << " )\n";
      cout << "src_addr ( " << event_ptr->src_addr.str_ip << ": " << event_ptr->src_addr.str_port << " )\n";
      cout << "loss_num: " << event_ptr->loss_num << endl;
      printf("----------------------------\n");

      break;
    }
    default:
    {
      printf("[RCV_DEFAULT_EVENT_TYPE]====\n");
      cout << "fd: " << event_ptr->fd << "\n";
      cout << "topic_name: " << event_ptr->topic_name << "\n";
      cout << "rcv_addr ( " << event_ptr->rcv_addr.str_ip << ": " << event_ptr->rcv_addr.str_port << " )\n";
      cout << "src_addr ( " << event_ptr->src_addr.str_ip << ": " << event_ptr->src_addr.str_port << " )" << endl;
      printf("----------------------------\n");

      break;
    }
  }

  return 0;
}

void* mws_event_dispatch_thread_routine(void* client_data)
{
  int rtv = g_evq_ptr->mws_event_dispatch();
  if (rtv < 0)
  {
    string log_body = "執行 mws_event_dispatch 失敗";
    cerr << log_body << endl;
    exit(1);
  }

  return NULL;
}

void show_usage()
{
  string str("");
  str = "Usage: mwsrcv CFG_FILE_PATH RCV'S_SECTION_NAME";
  cout << str << "\n";
  str = "       [-l MWS_LOG_FILE_PATH]";
  cout << str << "\n";
  str = "       [-c CTX'S_SECTION_NAME]";
  cout << str << "\n";
  str = "       [-e EVQ'S_SECTION_NAME]";
  cout << str << "\n";
  str = "       [-m MESSAGE(S)_PER_SECOND(0-10000)]";
  cout << str << "\n";
  str = "       [-s SIZE_OF_A_MESSAGE(1-1000)]";
  cout << str << "\n";
  str = "       [-v (show received message's HEX content)]";
  cout << str << "\n";
  str = "       [-t (show received message's text content)]";
  cout << str << endl;

  return;
}

void handle_argument(int argc, char** argv)
{
  // Begin: 必要的參數.
  if (argc >= 3)
  {
    g_mws_cfg_file_path = argv[1];
    g_rcv_section_name = argv[2];
  }
  // End: 必要的參數.

  // Begin: 選擇性參數.
  int opt = 0;
  const char* optstring = "l:c:e:m:s:vt";
  while ((opt = getopt_long(argc, argv, optstring, NULL, NULL)) != EOF)
  {
    switch (opt)
    {
      case 'l':
      {
        g_mws_log_file_path.assign(optarg, strlen(optarg));
        break;
      }
      case 'c':
      {
        g_ctx_section_name.assign(optarg, strlen(optarg));
        break;
      }
      case 'e':
      {
        g_evq_section_name.assign(optarg, strlen(optarg));
        break;
      }
      case 'm':
      {
        string str(optarg, strlen(optarg));

        #ifdef __TANDEM
          g_msgs_per_second = atoi(str.c_str());
        #else
          g_msgs_per_second = stoi(str, 0, 10);
        #endif

        break;
      }
      case 's':
      {
        string str(optarg, strlen(optarg));

        #ifdef __TANDEM
          g_size_of_a_msg = (int16_t)atoi(str.c_str());
        #else
          g_size_of_a_msg = (int16_t)stoi(str, 0, 10);
        #endif

        break;
      }
      case 'v':
      {
        g_show_msg_hex = true;
        break;
      }
      case 't':
      {
        g_show_msg_text = true;
        break;
      }
      case 'h':
      {
        show_usage();
        exit(0);
      }
      default:
      {
        show_usage();
        exit(1);
      }
    }
  }
  // End: 選擇性參數.

  // Begin: 檢查參數.
  if (check_argument() == false)
  {
    show_usage();
    exit(1);
  }
  // End: 檢查參數.

  return ;
}

bool check_argument()
{
  bool rtv = true;

  if (g_mws_cfg_file_path.size() <= 0)
  {
    cout << "CFG_FILE_PATH is required." << endl;
    rtv = false;
  }

  if (g_rcv_section_name.size() <= 0)
  {
    cout << "RCV'S_SECTION_NAME is required." << endl;
    rtv = false;
  }

  if ((g_msgs_per_second > 10000) || (g_msgs_per_second < 0))
  {
    cout << "MESSAGE(S)_PER_SECOND must be between 0 and 10000." << endl;
    rtv = false;
  }

  if ((g_size_of_a_msg > 1000) || (g_size_of_a_msg < 1))
  {
    cout << "SIZE_OF_A_MESSAGE must be between 1 and 1000." << endl;
    rtv = false;
  }

  return rtv;
}

void debug_show_argument()
{
  cout << "g_mws_cfg_file_path: " << g_mws_cfg_file_path << endl;
  cout << "g_rcv_section_name: " << g_rcv_section_name << endl;
  cout << "g_mws_log_file_path: " << g_mws_log_file_path << endl;
  cout << "g_ctx_section_name: " << g_ctx_section_name << endl;
  cout << "g_evq_section_name: " << g_evq_section_name << endl;
  cout << "g_msgs_per_second: " << g_msgs_per_second << endl;
  cout << "g_size_of_a_msg: " << g_size_of_a_msg << endl;
  if (g_show_msg_hex == true)
  {
    cout << "g_show_msg_hex: true"<< endl;
  }
  else
  {
    cout << "g_show_msg_hex: false"<< endl;
  }
  if (g_show_msg_text == true)
  {
    cout << "g_show_msg_text: true"<< endl;
  }
  else
  {
    cout << "g_show_msg_text: false"<< endl;
  }

  return;
}
