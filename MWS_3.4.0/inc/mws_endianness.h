// ら戳      セ       蝴臔    э.
// 20210420  v01.00.00  獵地      穝祘Α秨祇.

#ifndef MWS_ENDIANNESS_H_INCLUDED
#define MWS_ENDIANNESS_H_INCLUDED

#include <stdint.h>

// 弧
//   host: セ诀狠, ㄏノ big endian ( OSS) ┪ little endian ( x64  RHEL)
//   network: 呼隔肚块, ㄏノ big endian.

class mws_endianness;
typedef mws_endianness mws_endianness_t;

class mws_endianness
{
  public:
    // : 篶Α(constructor), ノ﹍てン.
    // 肚: 礚.
    // 把计: 礚.
    mws_endianness();
    // : 秆篶Α(destructor).
    // 肚: 礚.
    // 把计: 礚.
    ~mws_endianness();

    // : 眔 host 琌 big endian 吏挂.
    // 肚:
    //   true: host 琌 big endian 吏挂.
    //   false: host ぃ琌 big endian 吏挂.
    // 把计: 礚.
    bool is_big_endian_env();
    // : 眔 host 琌 little endian 吏挂.
    // 肚:
    //   true: host 琌 little endian 吏挂.
    //   false: host ぃ琌 little endian 吏挂.
    // 把计: 礚.
    bool is_little_endian_env();

    // : 盢 host  int16_t 锣传Θ network ノ int16_t.
    // 肚: network ノ int16_t.
    // 把计:
    //    i: host  int16_t.
    int16_t host_to_network_int16_t(int16_t i);
    // : 盢 host  int32_t 锣传Θ network ノ int32_t.
    // 肚: network ノ int32_t.
    // 把计:
    //    i: host  int32_t.
    int32_t host_to_network_int32_t(int32_t i);
    // : 盢 host  int64_t 锣传Θ network ノ int64_t.
    // 肚: network ノ int64_t.
    // 把计:
    //    i: host  int64_t.
    int64_t host_to_network_int64_t(int64_t i);
    // : 盢 host  uint16_t 锣传Θ network ノ uint16_t.
    // 肚: network ノ uint16_t.
    // 把计:
    //    i: host  uint16_t.
    uint16_t host_to_network_uint16_t(uint16_t i);
    // : 盢 host  uint32_t 锣传Θ network ノ uint32_t.
    // 肚: network ノ uint32_t.
    // 把计:
    //    i: host  uint32_t.
    uint32_t host_to_network_uint32_t(uint32_t i);
    // : 盢 host  uint64_t 锣传Θ network ノ uint64_t.
    // 肚: network ノ uint64_t.
    // 把计:
    //    i: host  uint64_t.
    uint64_t host_to_network_uint64_t(uint64_t i);

    // : 盢 host  int16_t 锣传Θ network ノ int16_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  int16_t, 挡琌 network ノ int16_t.
    void host_to_network_int16_t_ref(int16_t& i);
    // : 盢 host  int32_t 锣传Θ network ノ int32_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  int32_t, 挡琌 network ノ int32_t.
    void host_to_network_int32_t_ref(int32_t& i);
    // : 盢 host  int64_t 锣传Θ network ノ int64_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  int64_t, 挡琌 network ノ int64_t.
    void host_to_network_int64_t_ref(int64_t& i);
    // : 盢 host  uint16_t 锣传Θ network ノ uint16_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  uint16_t, 挡琌 network ノ uint16_t.
    void host_to_network_uint16_t_ref(uint16_t& i);
    // : 盢 host  uint32_t 锣传Θ network ノ uint32_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  uint32_t, 挡琌 network ノ uint32_t.
    void host_to_network_uint32_t_ref(uint32_t& i);
    // : 盢 host  uint64_t 锣传Θ network ノ uint64_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 host  uint64_t, 挡琌 network ノ uint64_t.
    void host_to_network_uint64_t_ref(uint64_t& i);

    // : 盢 network  int16_t 锣传Θ host ノ int16_t.
    // 肚: host ノ int16_t.
    // 把计:
    //    i: network  int16_t.
    int16_t network_to_host_int16_t(int16_t i);
    // : 盢 network  int32_t 锣传Θ host ノ int32_t.
    // 肚: host ノ int32_t.
    // 把计:
    //    i: network  int32_t.
    int32_t network_to_host_int32_t(int32_t i);
    // : 盢 network  int64_t 锣传Θ host ノ int64_t.
    // 肚: host ノ int64_t.
    // 把计:
    //    i: network  int64_t.
    int64_t network_to_host_int64_t(int64_t i);
    // : 盢 network  uint16_t 锣传Θ host ノ uint16_t.
    // 肚: host ノ uint16_t.
    // 把计:
    //    i: network  uint16_t.
    uint16_t network_to_host_uint16_t(uint16_t i);
    // : 盢 network  uint32_t 锣传Θ host ノ uint32_t.
    // 肚: host ノ uint32_t.
    // 把计:
    //    i: network  uint32_t.
    uint32_t network_to_host_uint32_t(uint32_t i);
    // : 盢 network  uint64_t 锣传Θ host ノ uint64_t.
    // 肚: host ノ uint64_t.
    // 把计:
    //    i: network  uint64_t.
    uint64_t network_to_host_uint64_t(uint64_t i);

    // : 盢 network  int16_t 锣传Θ host ノ int16_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  int16_t, 挡琌 host ノ int16_t.
    void network_to_host_int16_t_ref(int16_t& i);
    // : 盢 network  int32_t 锣传Θ host ノ int32_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  int32_t, 挡琌 host ノ int32_t.
    void network_to_host_int32_t_ref(int32_t& i);
    // : 盢 network  int64_t 锣传Θ host ノ int64_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  int64_t, 挡琌 host ノ int64_t.
    void network_to_host_int64_t_ref(int64_t& i);
    // : 盢 network  uint16_t 锣传Θ host ノ uint16_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  uint16_t, 挡琌 host ノ uint16_t.
    void network_to_host_uint16_t_ref(uint16_t& i);
    // : 盢 network  uint32_t 锣传Θ host ノ uint32_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  uint32_t, 挡琌 host ノ uint32_t.
    void network_to_host_uint32_t_ref(uint32_t& i);
    // : 盢 network  uint64_t 锣传Θ host ノ uint64_t.
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌 network  uint64_t, 挡琌 host ノ uint64_t.
    void network_to_host_uint64_t_ref(uint64_t& i);

    // : 盢 int16_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  int16_t.
    // 把计:
    //    i: 锣传玡 int16_t.
    int16_t convert_endian_int16_t(int16_t i);
    // : 盢 int32_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  int32_t.
    // 把计:
    //    i: 锣传玡 int32_t.
    int32_t convert_endian_int32_t(int32_t i);
    // : 盢 int64_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  int64_t.
    // 把计:
    //    i: 锣传玡 int64_t.
    int64_t convert_endian_int64_t(int64_t i);
    // : 盢 uint16_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  uint16_t.
    // 把计:
    //    i: 锣传玡 uint16_t.
    uint16_t convert_endian_uint16_t(uint16_t i);
    // : 盢 uint32_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  uint32_t.
    // 把计:
    //    i: 锣传玡 uint32_t.
    uint32_t convert_endian_uint32_t(uint32_t i);
    // : 盢 uint64_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 锣传筁 endian  uint64_t.
    // 把计:
    //    i: 锣传玡 uint64_t.
    uint64_t convert_endian_uint64_t(uint64_t i);

    // : 盢 int16_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 int16_t, 挡琌锣传筁 endian  int16_t.
    void convert_endian_int16_t_ref(int16_t& i);
    // : 盢 int32_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 int32_t, 挡琌锣传筁 endian  int32_t.
    void convert_endian_int32_t_ref(int32_t& i);
    // : 盢 int64_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 int64_t, 挡琌锣传筁 endian  int64_t.
    void convert_endian_int64_t_ref(int64_t& i);
    // : 盢 uint16_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 uint16_t, 挡琌锣传筁 endian  uint16_t.
    void convert_endian_uint16_t_ref(uint16_t& i);
    // : 盢 uint32_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 uint32_t, 挡琌锣传筁 endian  uint32_t.
    void convert_endian_uint32_t_ref(uint32_t& i);
    // : 盢 uint64_t  endian 锣传.
    //       (big endian -> little endian or little endian -> big endian).
    // 肚: 礚.
    // 把计:
    //    &i: 秨﹍琌锣传玡 uint64_t, 挡琌锣传筁 endian  uint64_t.
    void convert_endian_uint64_t_ref(uint64_t& i);

    // : debug ㄣ, 盢 *ptr ず甧ㄌΩ HEX よΑ.
    // 肚: 礚.
    // 把计:
    //    *ptr: 璶ず甧夹.
    //    len: 羆璶ぶ byte(s).
    void show_hex_value(const unsigned char* ptr, size_t len);

  private:
    // true: big endian environment,
    // false: little endian environment.
    bool is_big_endian;

};

#endif // MWS_ENDIANNESS_H_INCLUDED
