//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_LOG_CPP 1

#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#ifdef __TANDEM
  #include <ctime>
  #include <errno.h>
  // for Guardian file I/O.
  #include <cextdecs.h(FILE_SETKEY_, \
                       FILE_SETPOSITION_)>
#endif

#include "../inc/mws_global_variable.h"
#include "../inc/mws_log.h"
#include "../inc/mws_time.h"

using namespace std;

namespace mws_log
{
  static pthread_mutex_t g_pthread_mutex_object;
  static bool pthread_mutex_initialized_flag = false;
  mws_time_t time_obj;
  bool g_print_screen = false;
  std::string g_identity_name;

  // MWS log 郎隔畖.
  std::string g_log_file_path;

  #ifdef __TANDEM
    short file_num = -1;
  #else
    fstream mws_log_file;
  #endif

  std::string get_source_code_name(std::string path)
  {
    size_t found;
    found = path.find_last_of("/\\");

    // get source code name from origin path.
    std::string source_code_name = path.substr(found + 1);

    found = source_code_name.find_last_of(".");

    // return source code name without file extension.
    return source_code_name.substr(0, found);
  }

  // : initialize mws_log.
  // 肚 0: タ盽.
  //        -1: log file ぃ.
  // 把计 identity_name: AP 祘Α嘿 + class.
  // 把计 log_file_name: log 郎郎.
  int initialize_mws_log(std::string identity_name, const std::string log_file_name)
  {
    static bool is_called_initialize_mws_log = false;
    if (is_called_initialize_mws_log == false)
    {
      is_called_initialize_mws_log = true;

      g_identity_name = identity_name;

      g_log_file_path = log_file_name;

      if (pthread_mutex_initialized_flag == false)
      {
        pthread_mutex_init(&g_pthread_mutex_object, NULL);
        pthread_mutex_initialized_flag = true;
      }
    }

    #ifdef __TANDEM
      file_num = -1;
      short error = 0;
      // guardian source code name.
      unsigned short option = 0x0000;
      // oss source code name.
      //unsigned short option = 0x0020;

      // open guardian file.
      error = PUT_FILE_OPEN_((char*)g_log_file_path.c_str(),
                             (short)g_log_file_path.length(),
                             &file_num,
                             2, // write only.
                             0, // shared.
                             0, // await I/O.
                             ,
                             (short)option,
                             ,
                             ,
                             ,
                             1); // elections.
      if (error != 0)
      {
        g_print_screen = true;

        return -1;
      }

      PUT_FILE_CLOSE_(file_num);
    #else
      // 耞郎琌.
      if (access(g_log_file_path.c_str(), F_OK) != 0)
      {
        g_print_screen = true;

        return -1;
      }
    #endif

    return 0;
  }

  std::string get_free_format_log_string(Log_Format info)
  {
    std::stringstream log_stream;
    // field_name1: field_value1, field_name2:field_value2 Α块.
    log_stream << info.log_date
               << " " << info.log_time
               << " " << info.log_identity
               << "(" << info.log_topic_name
               << ") code:" << info.log_code
               << ", err:" << info.log_error_code
               << ", src:" << info.log_source_code
               << ", fun:" << info.log_function
               << ", line#:" << info.log_line_no
               << " " << info.log_body;

    if ((log_stream.str().length() % 2) == 0)
    {
      log_stream << " ";
    }
    log_stream << endl;

    return log_stream.str();
  }

  // 盢 log 糶郎.
  void write_log_line(Log_Format info)
  {
    // write a log line to target file.
    std::string log_line = "";

    // write a log line to file by free format.
    log_line = get_free_format_log_string(info);
    #ifdef __TANDEM
      //std::cout << "write_log_line: " << "this is NSK !!!" << std::endl;

      if (g_print_screen == false)
      {
        short error = FILE_SETPOSITION_(file_num, -1);
        if (error != 0)
        {
          cerr << "Failed to write free format log " << g_log_file_path.c_str() << endl;
          cerr << "FILE_SETPOSITION_ error: " << error << endl;
          cerr << "msg: " << log_line.c_str() << endl;
        }
        //else
        //{
        //  cerr << "FILE_SETPOSITION_ success: " << error << endl;
        //}
        error = PUT_WRITEX(file_num, log_line.c_str(), (int)log_line.length());
        if (error != 0)
        {
          cerr << "Failed to write free format log " << g_log_file_path.c_str() << endl;
          cerr << "PUT_WRITEX error: " << error << endl;
          cerr << "msg: " << log_line.c_str() << endl;
        }
        //else
        //{
        //  cerr << "PUT_WRITEX success: " << error << endl;
        //}
      }
      else
      {
        std::cout << log_line << std::endl;
      }
    #else
      //std::cout << "write_log_line: " << "this is Linux !!!" << std::endl;

      // write to log.
      if (g_print_screen == false)
      {
        mws_log_file << log_line << endl;
      }
      else
      {
        std::cout << log_line << std::endl;
      }
    #endif

    return ;
  }

  // : 盢 free format Α log 糶 log 郎.
  // 肚 void.
  // 把计 topic_name: topic name.
  // 把计 error_code: 岿粇絏.
  // 把计 code: E ボ岿粇, W ボ牡, N ボタ盽, Q ボ参璸戈.
  // 把计 source_code: ㊣ write_to_log ㄧΑ source code.
  // 把计 function: ㊣ write_to_log ㄧΑㄧΑ.
  // 把计 line_no: ㊣ write_to_log ㄧΑ︽腹.
  // 把计 body: パ恶﹃.
  void write_to_log(const std::string topic_name,
                    const int error_code,
                    const std::string code,
                    const std::string source_code,
                    const std::string function,
                    const int line_no,
                    std::string body)
  {
    pthread_mutex_lock(&g_pthread_mutex_object);
    // 秨 log 郎.
    #ifdef __TANDEM
      //std::cout << "write_to_log: " << "this is NSK !!!" << std::endl;

      //  append 家Α秨 LOG FILE.
      if (g_print_screen == false)
      {
        file_num = -1;
        short error = 0;
        // guardian source code name.
        unsigned short option = 0x0000;
        // oss source code name.
        //unsigned short option = 0x0020;

        // open guardian file.
        error = PUT_FILE_OPEN_((char*)g_log_file_path.c_str(),
                               (short)g_log_file_path.length(),
                               &file_num,
                               2, // write only.
                               0, // shared.
                               0, // await I/O.
                               ,
                               (short)option,
                               ,
                               ,
                               ,
                               1); // elections.
        if (error != 0)
        {
          cerr << "Failed to Open Log file " << g_log_file_path.c_str() << endl;
          cerr << "PUT_FILE_OPEN_() error: " << error << endl;
        }
        //else
        //{
        //  cerr << "Open Log file success: " << error << endl;
        //}
      }
    #else
      //std::cout << "write_to_log: " << "this is Linux !!!" << std::endl;

      //  append 家Α秨 LOG FILE.
      mws_log_file.clear();
      if (g_print_screen == false)
      {
        mws_log_file.open(g_log_file_path.c_str(), fstream::out | fstream::app);
        if (mws_log_file.fail())
        {
          cerr << "Failed to Open Log file " << g_log_file_path << endl;
          cerr << "failbit: " << (mws_log_file.rdstate() & fstream::failbit)
               << ", "
               << "badbit: " << (mws_log_file.rdstate() & fstream::badbit)
               << endl;
          cerr << "Please check MWS Log File." << endl;
        }
      }
    #endif

    // free format  log body ぃ.
    if (body.length() == 0)
    {
      body = " ";
    }

    Log_Format info;

    info.log_topic_name = topic_name;
    info.log_error_code = error_code;
    info.log_code = code;
    info.log_identity = g_identity_name;
    info.log_source_code = get_source_code_name(source_code);
    info.log_function = function;
    info.log_line_no = line_no;
    info.log_date = time_obj.get_local_date();
    info.log_time = time_obj.get_local_time();
    info.log_body = body;

    write_log_line(info);

    // 闽 log 郎.
    #ifdef __TANDEM
      PUT_FILE_CLOSE_(file_num);
    #else
      mws_log_file.close();
    #endif

    pthread_mutex_unlock(&g_pthread_mutex_object);

    return ;
  }
}
