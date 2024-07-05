// 日期      版本       維護人員    修改原因.
// 20210420  v01.00.00  吳青華      新程式開發.

//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_ENDIANNESS_CPP 1

#include <stdint.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "../inc/mws_endianness.h"

// 01 02 -> 02 01
#define CONVERT_ENDIAN_2_BYTES_SIGNED \
  i = (int16_t)(((uint16_t)i >> 8) | ((uint16_t)i << 8));

// 01 02 -> 02 01
#define CONVERT_ENDIAN_2_BYTES_UNSIGNED \
  i = (uint16_t)(((uint16_t)i >> 8) | ((uint16_t)i << 8));

// 01 02 03 04 -> 02 00 04 00 | 00 01 00 03 -> 02 01 04 03
// 02 01 04 03 -> 04 03 00 00 | 00 00 02 01 -> 04 03 02 01
#define CONVERT_ENDIAN_4_BYTES_SIGNED \
  uint32_t temp = ((((uint32_t)i << 8) & 0xFF00FF00) | \
                   (((uint32_t)i >> 8) & 0x00FF00FF)); \
  i = (int32_t)((temp << 16) | (temp >> 16));

// 01 02 03 04 -> 02 00 04 00 | 00 01 00 03 -> 02 01 04 03
// 02 01 04 03 -> 04 03 00 00 | 00 00 02 01 -> 04 03 02 01
#define CONVERT_ENDIAN_4_BYTES_UNSIGNED \
  uint32_t temp = ((((uint32_t)i << 8) & 0xFF00FF00) | \
                   (((uint32_t)i >> 8) & 0x00FF00FF)); \
  i = ((temp << 16) | (temp >> 16));

// 01 02 03 04 05 06 07 08 -> 05 06 07 08 00 00 00 00 | 00 00 00 00 01 02 03 04 ->
// 05 06 07 08 01 02 03 04
// 05 06 07 08 01 02 03 04 -> 07 08 00 00 03 04 00 00 | 00 00 05 06 00 00 01 02 ->
// 07 08 05 06 03 04 01 02
// 07 08 05 06 03 04 01 02 -> 08 00 06 00 04 00 02 00 | 00 07 00 05 00 03 00 01 ->
// 08 07 06 05 04 03 02 01
#define CONVERT_ENDIAN_8_BYTES_SIGNED \
    i = (int64_t)(((uint64_t)(i & 0x00000000FFFFFFFFull) << 32) | \
                  ((uint64_t)(i & 0xFFFFFFFF00000000ull) >> 32)); \
    i = (int64_t)(((uint64_t)(i & 0x0000FFFF0000FFFFull) << 16) | \
                  ((uint64_t)(i & 0xFFFF0000FFFF0000ull) >> 16)); \
    i = (int64_t)(((uint64_t)(i & 0x00FF00FF00FF00FFull) << 8)  | \
                  ((uint64_t)(i & 0xFF00FF00FF00FF00ull) >> 8));

// 01 02 03 04 05 06 07 08 -> 05 06 07 08 00 00 00 00 | 00 00 00 00 01 02 03 04 ->
// 05 06 07 08 01 02 03 04
// 05 06 07 08 01 02 03 04 -> 07 08 00 00 03 04 00 00 | 00 00 05 06 00 00 01 02 ->
// 07 08 05 06 03 04 01 02
// 07 08 05 06 03 04 01 02 -> 08 00 06 00 04 00 02 00 | 00 07 00 05 00 03 00 01 ->
// 08 07 06 05 04 03 02 01
#define CONVERT_ENDIAN_8_BYTES_UNSIGNED \
    i = (uint64_t)(((uint64_t)(i & 0x00000000FFFFFFFFull) << 32) | \
                  ((uint64_t)(i & 0xFFFFFFFF00000000ull) >> 32)); \
    i = (uint64_t)(((uint64_t)(i & 0x0000FFFF0000FFFFull) << 16) | \
                  ((uint64_t)(i & 0xFFFF0000FFFF0000ull) >> 16)); \
    i = (uint64_t)(((uint64_t)(i & 0x00FF00FF00FF00FFull) << 8)  | \
                  ((uint64_t)(i & 0xFF00FF00FF00FF00ull) >> 8));

mws_endianness::mws_endianness()
{
  uint16_t i = 1;
  char* ptr = (char*)&i;

  // 00 01 -> Big endain environment.
  if (ptr[0] == 0)
  {
    this->is_big_endian = true;
  }
  // 01 00 -> Little endain environment.
  else
  {
    this->is_big_endian = false;
  }

  return;
}

mws_endianness::~mws_endianness()
{

  return;
}

bool mws_endianness::is_big_endian_env()
{
  return this->is_big_endian;
}

bool mws_endianness::is_little_endian_env()
{
  return !(this->is_big_endian);
}

int16_t mws_endianness::host_to_network_int16_t(int16_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_SIGNED;
  }

  return i;
}

int32_t mws_endianness::host_to_network_int32_t(int32_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_SIGNED;
  }

  return i;
}

int64_t mws_endianness::host_to_network_int64_t(int64_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_SIGNED;
  }

  return i;
}

uint16_t mws_endianness::host_to_network_uint16_t(uint16_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_UNSIGNED;
  }

  return i;
}

uint32_t mws_endianness::host_to_network_uint32_t(uint32_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_UNSIGNED;
  }

  return i;
}

uint64_t mws_endianness::host_to_network_uint64_t(uint64_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_UNSIGNED;
  }

  return i;
}

void mws_endianness::host_to_network_int16_t_ref(int16_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::host_to_network_int32_t_ref(int32_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::host_to_network_int64_t_ref(int64_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::host_to_network_uint16_t_ref(uint16_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_UNSIGNED;
  }

  return;
}

void mws_endianness::host_to_network_uint32_t_ref(uint32_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_UNSIGNED;
  }

  return;
}

void mws_endianness::host_to_network_uint64_t_ref(uint64_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_UNSIGNED;
  }

  return;
}

int16_t mws_endianness::network_to_host_int16_t(int16_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_SIGNED;
  }

  return i;
}

int32_t mws_endianness::network_to_host_int32_t(int32_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_SIGNED;
  }

  return i;
}

int64_t mws_endianness::network_to_host_int64_t(int64_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_SIGNED;
  }

  return i;
}

uint16_t mws_endianness::network_to_host_uint16_t(uint16_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_UNSIGNED;
  }

  return i;
}

uint32_t mws_endianness::network_to_host_uint32_t(uint32_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_UNSIGNED;
  }

  return i;
}

uint64_t mws_endianness::network_to_host_uint64_t(uint64_t i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_UNSIGNED;
  }

  return i;
}

void mws_endianness::network_to_host_int16_t_ref(int16_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::network_to_host_int32_t_ref(int32_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::network_to_host_int64_t_ref(int64_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_SIGNED;
  }

  return;
}

void mws_endianness::network_to_host_uint16_t_ref(uint16_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_2_BYTES_UNSIGNED;
  }

  return;
}

void mws_endianness::network_to_host_uint32_t_ref(uint32_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_4_BYTES_UNSIGNED;
  }

  return;
}

void mws_endianness::network_to_host_uint64_t_ref(uint64_t& i)
{
  if (this->is_big_endian == false)
  {
    CONVERT_ENDIAN_8_BYTES_UNSIGNED;
  }

  return;
}

int16_t mws_endianness::convert_endian_int16_t(int16_t i)
{
  CONVERT_ENDIAN_2_BYTES_SIGNED;

  return i;
}

int32_t mws_endianness::convert_endian_int32_t(int32_t i)
{
  CONVERT_ENDIAN_4_BYTES_SIGNED;

  return i;
}

int64_t mws_endianness::convert_endian_int64_t(int64_t i)
{
  CONVERT_ENDIAN_8_BYTES_SIGNED;

  return i;
}

uint16_t mws_endianness::convert_endian_uint16_t(uint16_t i)
{
  CONVERT_ENDIAN_2_BYTES_UNSIGNED;

  return i;
}

uint32_t mws_endianness::convert_endian_uint32_t(uint32_t i)
{
  CONVERT_ENDIAN_4_BYTES_UNSIGNED;

  return i;
}

uint64_t mws_endianness::convert_endian_uint64_t(uint64_t i)
{
  CONVERT_ENDIAN_8_BYTES_UNSIGNED;

  return i;
}

void mws_endianness::convert_endian_int16_t_ref(int16_t& i)
{
  CONVERT_ENDIAN_2_BYTES_SIGNED;

  return;
}

void mws_endianness::convert_endian_int32_t_ref(int32_t& i)
{
  CONVERT_ENDIAN_4_BYTES_SIGNED;

  return;
}

void mws_endianness::convert_endian_int64_t_ref(int64_t& i)
{
  CONVERT_ENDIAN_8_BYTES_SIGNED;

  return;
}

void mws_endianness::convert_endian_uint16_t_ref(uint16_t& i)
{
  CONVERT_ENDIAN_2_BYTES_UNSIGNED;

  return;
}

void mws_endianness::convert_endian_uint32_t_ref(uint32_t& i)
{
  CONVERT_ENDIAN_4_BYTES_UNSIGNED;

  return;
}

void mws_endianness::convert_endian_uint64_t_ref(uint64_t& i)
{
  CONVERT_ENDIAN_8_BYTES_UNSIGNED;

  return;
}

void mws_endianness::show_hex_value(const unsigned char* ptr, size_t len)
{
  std::stringstream memory_data;

  for (size_t i = 0; i < len; ++i)
  {
    memory_data << std::hex
                << std::setw(2)
                << std::setfill ('0')
                << (unsigned short)(((ptr + i))[0])
                << " "
                << std::dec;
  }

  std::cout << memory_data.str() << std::endl;

  return;
}

