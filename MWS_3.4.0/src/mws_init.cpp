//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_INIT_CPP 1

#include <cstdint>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <signal.h>

#include "../inc/mws_debug.h"
#include "../inc/mws_init.h"
#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_time.h"
#include "../inc/mws_type_definition.h"

using namespace std;
using namespace mws_log;
using namespace mws_global_variable;

std::string trim_string(std::string str,
                        const bool convert_invisible_char_to_space_flag,
                        const bool trim_leading_spaces_flag,
                        const bool trim_trailing_spaces_flag);

int parse_INI_format_file(const std::string file_name,
                          std::string section_name,
                          std::map<std::string, std::string>& data_map);

int mws_init(const std::string identity_name,
             const std::string mws_cfg_file_path,
             const std::string mws_log_file_path,
             const int16_t mws_log_level);

std::string mws_get_error_msg(uint32_t mws_error_number);

std::map<std::string, std::map<std::string, std::string> > mws_get_cfg();


// 功能: 初始化 mws.
//       1. 初始化全域變數.
//       2. 載入 config file.
//       3. 設定 log file 位置.
// 回傳值 0: 初始化完成.
//        1:  已經做過初始化, 本次呼叫無效.
//        -1: 初始化失敗.
// 參數 identity_name: program name + class number.
// 參數 mws_cfg_file_path: config file 的位置.
// 參數 mws_log_file_path: log file 的位置.
// 參數 mws_log_level: 0 表示只寫必須的 log 訊息.
//                     1 表示寫入 debug 用的訊息.
int mws_init(const std::string identity_name,
             const std::string mws_cfg_file_path,
             const std::string mws_log_file_path,
             const int16_t mws_log_level)
{
  // mws_init() 只需要被呼叫一次.
  static bool is_loaded = false;

  if (is_loaded == true)
  {
    // 已經做過初始化, 本次呼叫無效.
    return 1;
  }
  else
  {
    is_loaded = true;
  }

  // Ignore the signal SIGPIPE.
  signal(SIGPIPE, SIG_IGN);

  // Begin: 初始化全域變數.

  // 初始化 g_cfg_file_path.
  g_cfg_file_path = mws_cfg_file_path;

  // Begin: 初始化 g_config_mapping.
  {
    // 開啟 mws config 檔取得 section name (AP 不保證提供 AP config).
    ifstream ifs(g_cfg_file_path.c_str(), ifstream::in);

    // ini file is not opened.
    if (!ifs.good())
    {
      std::string log_body;
      log_body = "config file (" + g_cfg_file_path + ") is not opened";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      return -1;
    }

    std::string temp_string("");
    std::getline(ifs, temp_string);
    // read file line by line.
    while (ifs.good())
    {
      temp_string = trim_string(temp_string, true, true, true);

      // 確保讀到的那行不是註解.
      if (!((temp_string[0] == ';') || (temp_string[0] == '#')))
      {
        // 如果是 section name 才呼叫 parse_INI_format_file.
        if ((temp_string[0] == '[') && (temp_string[ temp_string.size() - 1 ] == ']'))
        {
          std::string section_name = temp_string.substr(1, (temp_string.size() - 2) );
          std::map<std::string, std::string> data_map;
          // 根據 section name 讀取設定.
          int rtv = parse_INI_format_file(g_cfg_file_path,
                                          section_name,
                                          data_map);
          if (rtv == -1)
          {
            std::string log_body;
            log_body = "config file (" + g_cfg_file_path + ") is not opened";
            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
          }
          if (rtv == -2)
          {
            std::string log_body;
            log_body = "fail to parse section (" + section_name + ") in config file (" + g_cfg_file_path + ")";
            write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
          }
          // 將以 section name 為 key, name-value pair 的 data_map 為 value 的資料放到 g_config_mapping.
          g_config_mapping[section_name] = data_map;
        }
      }
      // 讀下一行.
      std::getline(ifs, temp_string);
    }

    if (ifs.is_open())
    {
      // 關閉 mws config 檔.
      ifs.close();
    }
  }
  // End: 初始化 g_config_mapping.

  // 初始化 mws_global_variable::g_mws_log_file_path.
  g_mws_log_file_path = mws_log_file_path;

  //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " g_mws_log_file_path:" << g_mws_log_file_path << std::endl;

  // Begin: 初始化 g_mws_log_level.
  {
    g_mws_log_level = mws_log_level;
    if ((g_mws_log_level != 0) &&
        (g_mws_log_level != 1))
    {
      g_mws_log_level = 0;
    }
  }
  // End: 初始化 g_mws_log_level.

  // 初始化 g_num_of_ctx.
  g_num_of_ctx = 0;

  // 初始化 g_num_of_evq.
  g_num_of_evq = 0;

  // Begin: 初始化 g_mws_global_mutex.
  {
    int rtv = pthread_mutex_init(&g_mws_global_mutex, NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(g_mws_global_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      return -1;
    }
  }
  // End: 初始化 g_mws_global_mutex.

  // Begin: 初始化 g_fd_table.
  {
    sess_addr_pair_t default_rcv_connection_setting;
    //sockaddr_in_t default_sockaddr_in_t;
    for (int i = 0; i < MAX_FD_SIZE; ++i)
    {
      int rtv = pthread_mutex_init(&(g_fd_table[i].mutex), NULL);
      if (rtv != 0)
      {
        std::string log_body;
        log_body = "pthread_mutex_init failed. rtv: ";
        log_body += std::to_string(rtv);
        write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

        return -1;
      }

      g_fd_table[i].fd_init(false);
    } // for (int i = 0; i < MAX_FD_SIZE; ++i)
  }
  // End: 初始化 g_fd_table.

  // 初始化 g_size_of_mws_pkg_head.
  g_size_of_mws_pkg_head = sizeof(mws_pkg_head_t);
  // debug.
  //cout << "g_size_of_mws_pkg_head: " << g_size_of_mws_pkg_head << endl;

  // 初始化 g_size_of_mws_pkg_t.
  g_size_of_mws_pkg_t = sizeof(mws_pkg_t);

  // 初始化 g_size_of_mws_topic_name_msg_t.
  g_size_of_mws_topic_name_msg_t = sizeof(mws_topic_name_msg_t);
  // debug.
  //cout << "g_size_of_mws_topic_name_msg_t: " << g_size_of_mws_topic_name_msg_t << endl;

  // End: 初始化全域變數.

  // Begin: 初始化 NWS log.
  {
    //std::cout << std::string(__func__) << ":" << std::to_string(__LINE__ ) << " g_mws_log_file_path:" << g_mws_log_file_path << std::endl;

    // 因為 mws log file 可以不存在, 故不存在時刷 warning 訊息.
    int rtv = initialize_mws_log(identity_name, g_mws_log_file_path);
    if (rtv == -1)
    {
      std::string log_body;
      log_body = "log file (" + g_mws_log_file_path + ") does not exist";
      write_to_log("", 1, "W", __FILE__, __func__, __LINE__, log_body);
    }
  }
  // End: 初始化 NWS log.

  // 初始化 g_time_current.
  g_time_current = time(NULL);
  // Begin: 初始化 g_time_current_mutex.
  {
    int rtv = pthread_mutex_init(&g_time_current_mutex, NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(g_time_current_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      return -1;
    }
  }
  // End: 初始化 g_time_current_mutex.

  // Begin: 初始化 g_mws_debug_log_mutex.
  #if (MWS_DEBUG == 1)
  {
    int rtv = pthread_mutex_init(&g_mws_debug_log_mutex, NULL);
    if (rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_mutex_init(g_mws_debug_log_mutex) failed. rtv: ";
      log_body += std::to_string(rtv);
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      return -1;
    }

    int pthread_rtv = pthread_create(&g_mws_debug_thread_id, NULL, mws_debug_fun, NULL);
    if (pthread_rtv != 0)
    {
      std::string log_body;
      log_body = "pthread_create() failed (rtv: " +
                 std::to_string(pthread_rtv) +
                 ", errno: " + std::to_string(errno) +
                 ", strerr: " + strerror(errno) + ")";
      write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

      return -1;
    }
  }
  #endif
  // End: 初始化 g_mws_debug_log_mutex.

  return 0;
}

// 功能: 讀取 ini 檔案格式內的資料.
//       如果 section_name 內容為空, 則讀取所有檔案內 name-value pair 資料到 map.
//       如果 section_name 內容不為空, 則讀取檔案內屬於該 section 的 name-value pair 資料到 map.
// 回傳值 0: parse 完成.
//        -1: 讀 ini 檔失敗.
//        -2: parse 失敗.
// 參數 file_name: ini 檔案路徑和檔名.
//      section_name: 所指定的 section name.
//      data_map: 存放 name-value pair 資料的 map.
int parse_INI_format_file(const std::string file_name,
                          std::string section_name,
                          std::map<std::string, std::string>& data_map)
{
  try
  {
    // 開啟 config 檔取得對應 section 的 name-value pair.
    ifstream ifs(file_name.c_str(), ifstream::in);

    // ini file is not opened.
    if (!ifs.good())
    {
      return -1;
    }

    // 調整 section_name 為 [section_name].
    section_name = trim_string(section_name, true, true, true);
    section_name = "[" + section_name + "]";

    std::string temp_string("");
    std::getline(ifs, temp_string);
    if (ifs.good())
    {
      temp_string = trim_string(temp_string, true, true, true);

      // 尋找指定的 section.
      while (temp_string != section_name)
      {
        std::getline(ifs, temp_string);
        if (!ifs.good())
        {
          // 關閉 config 檔.
          ifs.close();
          return -2;
        }
        temp_string = trim_string(temp_string, true, true, true);
      }
    }
    else
    {
      // 關閉 config 檔.
      ifs.close();
      return -2;
    }

    // 找指定的 section 內的資料.
    std::getline(ifs, temp_string);
    // read file line by line.
    while (ifs.good())
    {
      temp_string = trim_string(temp_string, true, true, true);

      // 如果不是註解也不是下一個 section 才取 name-value pair 放到 map.
      if ((!((temp_string[0] == ';') || (temp_string[0] == '#'))) &&
          (!((temp_string[0] == '[') && (temp_string[temp_string.length() - 1] == ']'))))
      {
        size_t found_position = temp_string.find("=");
        // 確保有 value 後將 name-value pair 放到 map.
        if (found_position != std::string::npos)
        {
          data_map[temp_string.substr(0, found_position)] = temp_string.substr(found_position + 1);
        }
      }
      // 碰到下一個 section 代表當前 section parse 完成.
      else if ((temp_string[0] == '[') &&
               (temp_string[temp_string.length() - 1] == ']'))
      {
        // 關閉 config 檔.
        ifs.close();
        return 0;
      }
      // 讀下一行.
      std::getline(ifs, temp_string);
    }
    // 讀最後一行.
    if (ifs.eof())
    {
      // 如果不是註解也不是下一個 section 才取 name-value pair 放到 map.
      if ((!((temp_string[0] == ';') || (temp_string[0] == '#'))) &&
          (!((temp_string[0] == '[') && (temp_string[temp_string.length() - 1] == ']'))))
      {
        size_t found_position = temp_string.find("=");
        // 確保有 value 後將 name-value pair 放到 map.
        if (found_position != std::string::npos)
        {
          data_map[temp_string.substr(0, found_position)] = temp_string.substr(found_position + 1);
        }
      }
    }

    // 關閉 config 檔.
    ifs.close();
    return 0;
  }
  catch (std::exception& e)
  {
    mws_time_t time_obj;
    std::cerr << "date: " << time_obj.get_local_date() << ", "
              << "time: " << time_obj.get_local_time() << ", "
              << "file: " << __FILE__ << ", "
              << "func: " << __func__ << ", "
              << "line: " << __LINE__ << ", "
              << "exception: " << e.what()
              << std::endl;
    return -1;
  }
}

// 功能: 將 string 的 invisible character 轉換成 space.
//       將 string 的 leading spaces 去除.
//       將 string 的 trailing spaces 去除.
// 回傳值: 經過所選擇的功能整理過後的 string.
// 參數 str: 要轉換的字串.
//      convert_invisible_char_to_space_flag: true 為要執行功能1, false 則不執行.
//      trim_leading_spaces_flag: true 為要執行功能2, false 則不執行.
//      trim_trailing_spaces_flag: true 為要執行功能3, false 則不執行.
std::string trim_string(std::string str,
                        const bool convert_invisible_char_to_space_flag,
                        const bool trim_leading_spaces_flag,
                        const bool trim_trailing_spaces_flag)
{
  try
  {
    if (convert_invisible_char_to_space_flag == true)
    {
      for (unsigned int i = 0; i < str.length(); ++i)
      {
        // ASCII Code 中 32 - 126 為 visible character.
        // 將非 invisible character 轉成 space(32).
        if ((str[i] < 32) || (str[i] > 126))
        {
          str[i] = 32;
        }
      }
    }
    if (trim_leading_spaces_flag == true)
    {
      size_t startpos = str.find_first_not_of(" \t");
      if (string::npos != startpos)
      {
        str = str.substr(startpos);
      }
    }
    if (trim_trailing_spaces_flag == true)
    {
      size_t endpos = str.find_last_not_of(" \t");
      if (string::npos != endpos)
      {
        str = str.substr(0, endpos+1);
      }
    }

    return str;
  }
  catch (std::exception& e)
  {
    mws_time_t time_obj;
    std::cerr << "date: " << time_obj.get_local_date() << ", "
              << "time: " << time_obj.get_local_time() << ", "
              << "file: " << __FILE__ << ", "
              << "func: " << __func__ << ", "
              << "line: " << __LINE__ << ", "
              << "exception: " << e.what()
              << std::endl;
    return "";
  }
}

// 功能: 將 mws 的 error number 轉換成文字說明.
// 回傳值: mws 的 error number 對應的文字說明.
// 參數 mws_error_number: mws 的 error number.
std::string mws_get_error_msg(uint32_t mws_error_number)
{
  std::string err_msg("Unknown error number");

  switch (mws_error_number)
  {
    case MWS_NO_ERROR:
    {
      err_msg = "No error";
      break;
    }
    case MWS_ERROR_PTHREAD_CREATE:
    {
      err_msg = "Error during pthread creation";
      break;
    }
    case MWS_ERROR_TOPIC_NAME:
    {
      err_msg = "Error during checking topic name";
      break;
    }
    case MWS_ERROR_LISTEN_SOCKET_CREATE:
    {
      err_msg = "Error during listen socket creation";
      break;
    }
    case MWS_ERROR_CONNECT_SOCKET_CREATE:
    {
      err_msg = "Error during connect socket creation";
      break;
    }
    default:
    {
      err_msg = "Unknown error number";
      break;
    }
  }

  return err_msg;
}

// 功能: 回傳所有的設定值.
// 回傳值: 所有的設定值.
// 參數: 無.
std::map<std::string, std::map<std::string, std::string> > mws_get_cfg()
{
  std::map<std::string, std::map<std::string, std::string> > config_mapping;
  config_mapping = g_config_mapping;

  return config_mapping;
}

#if (MWS_DEBUG == 1)
  void g_mws_global_mutex_lock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " bf lock g_mws_global_mutex";
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    pthread_mutex_lock(&g_mws_global_mutex);

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " locking g_mws_global_mutex";
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }

  int g_mws_global_mutex_trylock(const std::string file, const std::string function, const int line_no)
  {
    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " try lock g_mws_global_mutex";
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return pthread_mutex_trylock(&g_mws_global_mutex);
  }

  void g_mws_global_mutex_unlock(const std::string file, const std::string function, const int line_no)
  {
    pthread_mutex_unlock(&g_mws_global_mutex);

    {
      std::string log;
      log += file;
      log += " ";
      log += function;
      log += " ";
      log += std::to_string(line_no);
      log += " unlock g_mws_global_mutex";
      pthread_mutex_lock(&g_mws_debug_log_mutex);
      g_mws_debug_log.push_back(log);
      pthread_mutex_unlock(&g_mws_debug_log_mutex);
    }

    return;
  }
#else
  void g_mws_global_mutex_lock()
  {
    pthread_mutex_lock(&g_mws_global_mutex);
    return;
  }

  int g_mws_global_mutex_trylock()
  {
    return pthread_mutex_trylock(&g_mws_global_mutex);
  }

  void g_mws_global_mutex_unlock()
  {
    pthread_mutex_unlock(&g_mws_global_mutex);
    return;
  }
#endif
