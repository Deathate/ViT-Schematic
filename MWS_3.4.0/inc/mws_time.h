// ら戳      セ       蝴臔    э.
// 20130201  v01.00.00  獵地      穝祘Α秨祇.
// 20150120  v01.01.00  獵地      穝糤"耞琌瞷丁琌单把计丁"ㄧΑ.

#ifndef MWS_TIME_H_
#define MWS_TIME_H_

#include <string>
#include <ctime>
#include <sys/time.h>

class mws_time;
typedef mws_time mws_time_t;

class mws_time
{
  public:
    // : 篶Α(constructor), ノ﹍てン.
    // 肚: 礚.
    // 把计: 礚.
    mws_time();
    // : 秆篶Α(destructor).
    // 肚: 礚.
    // 把计: 礚.
    ~mws_time();

    // : gmtime()  thread safe .
    // 肚 void
    // 把计 timer: time_t 丁, 璶盢丁锣Θ UTC tm Α丁.
    // 把计 result: struct tm Α local 跋丁.
    void gmtime_TS(const time_t &timer,
                   struct tm &result);

    // : localtime()  thread safe .
    // 肚 void
    // 把计 timer: time_t 丁, 璶盢丁锣Θ local 跋丁.
    // 把计 result: struct tm Α local 跋丁.
    void localtime_TS(const time_t &timer,
                      struct tm &result);

    // : 眔 local ら戳.
    // 肚: std::string 篈 local ら戳.
    std::string get_local_date();

    // : 眔讽瞷ら戳, Α YYYYMMDD
    // 肚: Α YYYYMMDD 讽瞷ら戳.
    // ⊿Τ把计.
    unsigned long long int get_current_local_date();

    // : 眔 local 丁.
    // 肚: std::string 篈 local 丁.
    std::string get_local_time();

    // : 眔讽瞷丁, Αㄢㄢだㄢ millisecond  microsecond
    // 肚: Αㄢㄢだㄢ millisecond  microsecond 讽瞷丁.
    // ⊿Τ把计.
    unsigned long long int get_current_local_time();

    // : 眔 local 丁, Α microsecond
    // 肚: std::string 篈 local 丁.
    std::string get_local_time_microsecond();

    // : 眔讽瞷丁, Α microsecond
    // 肚: Α microsecond 讽瞷丁.
    // ⊿Τ把计.
    unsigned long long int get_current_local_time_microsecond();

    // : 盢 timeval 锣传Θ YYYYMMDD (ら戳)计.
    // 肚: YYYYMMDD  unsigned long long int
    // 把计 tim: 璶锣传 timeval Α丁.
    unsigned long long int timeval_to_local_date(const timeval &tim);

    // : 盢 timeval 锣传Θㄢㄢだㄢ millisecond  microsecond 计.
    // 肚: ㄢㄢだㄢ millisecond  microsecond  unsigned long long int
    // 把计 tim: 璶锣传 timeval Α丁.
    unsigned long long int timeval_to_local_time(const timeval &tim);

    // : 耞琌瞷丁琌单把计丁.
    // 肚: true ボ瞷丁单把计丁, false ボ瞷丁把计丁.
    // 把计 hour: 碭翴.
    // 把计 min: 碭だ.
    bool current_time_is_euqal_to_or_later_than(const int hour,
                                                const int min);

    // : 耞琌瞷丁琌单把计丁.
    // 肚: true ボ瞷丁单把计丁, false ボ瞷丁把计丁.
    // 把计 hour: 碭翴.
    // 把计 min: 碭だ.
    bool current_time_is_euqal_to_or_earlier_than(const int hour,
                                                  const int min);

    // : 秈セㄧΑ, 礛 delay 把计丁瞒秨.
    // 肚 void
    // 把计 hour: 碭翴.
    // 把计 min: 碭だ.
    // 把计 second: 碭.
    void delay_to_input_time(const int hour,
                             const int min,
                             const int second);
};

#endif // MWS_TIME_H_
