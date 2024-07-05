#ifndef MWS_LOG_H_
#define MWS_LOG_H_

#include <string>
#include <vector>

// 弧: log_body パ﹃Α log 癘魁よΑ, log_body 璶Τ戈.

namespace mws_log
{
  typedef struct _Log_Format
  {
    std::string log_identity;
    std::string log_topic_name;
    int log_error_code;
    std::string log_code;
    std::string log_source_code;
    std::string log_function;
    int log_line_no;
    std::string log_date;
    std::string log_time;
    std::string log_body;
  } Log_Format;

  // : initialize mws_log.
  // 肚 0: タ盽.
  //        -1: log file ぃ.
  // 把计 identity_name: AP 祘Α嘿 + class.
  // 把计 log_file_name: log 郎郎.
  int initialize_mws_log(std::string identity_name, const std::string log_file_name);

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
                    std::string body);
}

#endif // MWS_LOG_H_
