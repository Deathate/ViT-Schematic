#ifndef MWS_CFG_H_
#define MWS_CFG_H_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

// : ﹍て mws
//       1. 更 config file.
//       2. 砞﹚ log file 竚.
// 肚 0: ﹍てЧΘ.
//        1:  竒暗筁﹍て, セΩ㊣礚.
//        -1: ﹍てア毖.
// 把计 identity_name: program name + class number.
// 把计 mws_cfg_file_path: config file 竚.
// 把计 mws_log_file_path: log file 竚.
// 把计 mws_log_level: 0 ボ糶ゲ斗 log 癟.
//                     1 ボ糶 debug ノ癟.
int mws_init(const std::string identity_name,
             const std::string mws_cfg_file_path,
             const std::string mws_log_file_path,
             const int16_t mws_log_level = 1);

// : 盢 mws  error number 锣传Θゅ弧.
// 肚: mws  error number 癸莱ゅ弧.
// 把计 mws_error_number: mws  error number.
std::string mws_get_error_msg(uint32_t mws_error_number);

// : 肚┮Τ砞﹚.
// 肚: ┮Τ砞﹚.
// 把计: 礚.
std::map<std::string, std::map<std::string, std::string> > mws_get_cfg();

#if (MWS_DEBUG == 1)
  void g_mws_global_mutex_lock(const std::string file, const std::string function, const int line_no);
  int g_mws_global_mutex_trylock(const std::string file, const std::string function, const int line_no);
  void g_mws_global_mutex_unlock(const std::string file, const std::string function, const int line_no);
#else
  void g_mws_global_mutex_lock();
  int g_mws_global_mutex_trylock();
  void g_mws_global_mutex_unlock();
#endif

#endif /* MWS_CFG_H_ */
