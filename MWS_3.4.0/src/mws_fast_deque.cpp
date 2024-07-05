// 日期      版本       維護人員    修改原因.
// 20200206  v01.00.00  吳青華      新程式開發.
// 20220908  v01.01.00  吳青華      pop_all() 判斷若 deque 沒有資料直接 return 0.

//////////////////////////////////////////////////////////////////////
// Define Declaration
//////////////////////////////////////////////////////////////////////
#define MWS_FAST_DEQUE_CPP 1

#include <sys/types.h>

#include <stdint.h>

#include <cstdlib>
#include <cstring>

#include <iostream>
#include <vector>

// debug
#include <stdio.h>

#include "../inc/mws_fast_deque.h"
#include "../inc/mws_log.h"

#define MIN_DEFAULT_NODE_QUANTITY 100
#define MIN_EXTENDED_NODE_QUANTITY 10

namespace mws_fast_deque_global_variables
{
  static const ssize_t gc_sizeof_ptr = sizeof(void*);
  static const ssize_t gc_sizeof_2_ptr = 2 * sizeof(void*);
}

using namespace mws_log;
using namespace mws_fast_deque_global_variables;

mws_fast_deque::mws_fast_deque(ssize_t data_size,
                              ssize_t default_node_qty,
                              ssize_t extended_node_qty)
{
  this->data_size = data_size;
  if (default_node_qty < MIN_DEFAULT_NODE_QUANTITY)
  {
    default_node_qty = MIN_DEFAULT_NODE_QUANTITY;
  }
  if (extended_node_qty < MIN_EXTENDED_NODE_QUANTITY)
  {
    extended_node_qty = MIN_EXTENDED_NODE_QUANTITY;
  }
  this->pool_ptr = (char*)calloc(default_node_qty, (data_size + gc_sizeof_2_ptr));
  this->block.push_back(this->pool_ptr);
  this->number_of_nodes_in_pool = default_node_qty;
  this->number_of_node_in_new_block = extended_node_qty;
  this->gc_ptr = NULL;
  this->next_deque_no = 0;

  int rtv = pthread_mutex_init(&(this->mutex), NULL);
  if (rtv != 0)
  {
    std::string log_body = "pthread_mutex_init() failed (rtv: " + std::to_string(rtv)
                           + ", errno: " + std::to_string(errno)
                           + ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);

    exit(1);
  }

  return;
}

mws_fast_deque::~mws_fast_deque()
{
  // 將 allocate 到的記憶體釋放.
  for (size_t i = 0; i < this->block.size(); ++i)
  {
    free(this->block[i]);
  }

  this->block.resize(0);

  int rtv = pthread_mutex_destroy(&(this->mutex));
  if (rtv != 0)
  {
    std::string log_body = "pthread_mutex_destroy() failed (rtv: " + std::to_string(rtv)
                           + ", errno: " + std::to_string(errno)
                           + ", strerr: " + strerror(errno) + ")";
    write_to_log("", -1, "E", __FILE__, __func__, __LINE__, log_body);
  }

  return;
}

ssize_t mws_fast_deque::get_data_size()
{
  return this->data_size;
}

ssize_t mws_fast_deque::get_new_deque()
{
  this->deque_head.push_back(NULL);
  this->deque_tail.push_back(NULL);
  this->deque_current.push_back(NULL);
  this->deque_curr_pos.push_back(0);
  this->deque_at.push_back(NULL);
  this->deque_at_pos.push_back(0);
  this->deque_size.push_back(0);

  return this->next_deque_no++;
}

char* mws_fast_deque::push_back_a_black_block(ssize_t deque_no)
{
  // 不存在這個 deque.
  //if (deque_no >= this->next_deque_no)
  //{
  //  return -1;
  //}

  char* new_node = this->get_a_node();

  // 如果 dequeu 已經存在 node.
  if (this->deque_tail[deque_no] != NULL)
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 tail node.
    //printf("this->deque_tail[%d] = %p\n", deque_no, this->deque_tail[deque_no]);
    *(void**)(new_node + this->data_size) = this->deque_tail[deque_no];

    // 2. 維護 new node 指向 next node 的指標: 指向 NULL.
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = NULL;

    // 3. 維護 tail node 指向 next node 的指標: 指向 new node.
    *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = new_node;

    // 4. 將 new node 變成新的 tail node.
    this->deque_tail[deque_no] = new_node;
  }
  // 如果 deque 沒有任何 node.
  else
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 NULL.
    //char* previous_node_ptr = new_node + this->data_size;
    //*(void**)previous_node_ptr = NULL;
    *(void**)(new_node + this->data_size) = NULL;

    // 2. 維護 new node 指向 next node 的指標: 指向 NULL.
    //char* next_node_ptr = new_node + this->data_size + gc_sizeof_ptr;
    //*(void**)next_node_ptr = NULL;
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = NULL;

    // 3. 將 head 指向第一個 node.
    this->deque_head[deque_no] = new_node;

    // 4. 將 tail 指向第一個 node.
    this->deque_tail[deque_no] = new_node;
  }

  // deque 的 size 增加 1.
  ++this->deque_size[deque_no];

  return new_node;
}

int16_t mws_fast_deque::push_back(ssize_t deque_no, void* data_ptr)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  char* new_node = this->get_a_node();
  // 將資料放到 new node.
  memcpy((void*)new_node, data_ptr, this->data_size);

  // 如果 dequeu 已經存在 node.
  if (this->deque_tail[deque_no] != NULL)
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 tail node.
    //printf("this->deque_tail[%d] = %p\n", deque_no, this->deque_tail[deque_no]);
    *(void**)(new_node + this->data_size) = this->deque_tail[deque_no];

    // 2. 維護 new node 指向 next node 的指標: 指向 NULL.
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = NULL;

    // 3. 維護 tail node 指向 next node 的指標: 指向 new node.
    *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = new_node;

    // 4. 將 new node 變成新的 tail node.
    this->deque_tail[deque_no] = new_node;
  }
  // 如果 deque 沒有任何 node.
  else
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 NULL.
    //char* previous_node_ptr = new_node + this->data_size;
    //*(void**)previous_node_ptr = NULL;
    *(void**)(new_node + this->data_size) = NULL;

    // 2. 維護 new node 指向 next node 的指標: 指向 NULL.
    //char* next_node_ptr = new_node + this->data_size + gc_sizeof_ptr;
    //*(void**)next_node_ptr = NULL;
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = NULL;

    // 3. 將 head 指向第一個 node.
    this->deque_head[deque_no] = new_node;

    // 4. 將 tail 指向第一個 node.
    this->deque_tail[deque_no] = new_node;
  }

  // deque 的 size 增加 1.
  ++this->deque_size[deque_no];

  return 0;
}

int16_t mws_fast_deque::push_front(ssize_t deque_no, void* data_ptr)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  char* new_node = this->get_a_node();
  // 將資料放到 new node.
  memcpy((void*)new_node, data_ptr, this->data_size);

  // 如果 dequeu 已經存在 node.
  if (this->deque_head[deque_no] != NULL)
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 null node. (new->p 指向 NULL)
    *(void**)(new_node + this->data_size) = NULL;

    // 2. 維護 new node 指向 next node 的指標: 指向 head node. (new->n 指向 head)
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = this->deque_head[deque_no];

    // 3. 維護 head node 指向 previous node 的指標: 指向 new node.
    *(void**)(this->deque_head[deque_no] + this->data_size) = new_node;

    // 4. 將 new node 變成新的 head node. (h 指向 new)
    this->deque_head[deque_no] = new_node;
  }
  // 如果 deque 沒有任何 node.
  else
  {
    // 1. 維護 new node 指向 previous node 的指標: 指向 NULL.
    *(void**)(new_node + this->data_size) = NULL;

    // 2. 維護 new node 指向 next node 的指標: 指向 NULL.
    *(void**)(new_node + this->data_size + gc_sizeof_ptr) = NULL;

    // 3. 將 head 指向第一個 node.
    this->deque_head[deque_no] = new_node;

    // 4. 將 tail 指向第一個 node.
    this->deque_tail[deque_no] = new_node;
  }

  // deque 的 size 增加 1.
  ++this->deque_size[deque_no];

  // 新增一筆資料在最前面, 且 deque_curr_pos 非 0, 則 deque_curr_pos 需加 1.
  // 但 deque_curr_pos 不需要動.
  if (this->deque_curr_pos[deque_no] > 0)
  {
    this->deque_curr_pos[deque_no] += 1;
  }

  // 新增一筆資料在最前面, 且 this->deque_at 非 0, 則 this->deque_at 需加 1.
  // 但 deque_at_pos 不需要動.
  if (this->deque_at_pos[deque_no] > 0)
  {
    this->deque_at_pos[deque_no] += 1;
  }

  return 0;
}

int16_t mws_fast_deque::pop_back(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // deque 裡面沒有資料.
  if (this->deque_head[deque_no] == NULL)
  {
    return 1;
  }

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // deque 除了這一個要 pop 的 node 還有其他 node 存在.
    if (*(void**)(this->deque_tail[deque_no] + this->data_size) != NULL)
    {
      // 1. gc->p 指向 t.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_tail[deque_no];
      // 2. t->n 指向 gc.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 3. gc 指向 t.
      this->gc_ptr = this->deque_tail[deque_no];
      // 4. t 指向 t->p
      this->deque_tail[deque_no] = (char*)*(void**)(this->deque_tail[deque_no] + this->data_size);
      // 5. gc->p 指向 NULL.
      *(void**)(this->gc_ptr + this->data_size) = NULL;
      // 6. t->n 指向 NULL.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = NULL;
    }
    // deque 只剩這一個要 pop 的 node 存在.
    else
    {
      // 1. gc->p 指向 t.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_tail[deque_no];
      // 2. t->n 指向 gc.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 3. gc 指向 t.
      this->gc_ptr = this->deque_tail[deque_no];
      // 4. h 指向 NULL.
      this->deque_head[deque_no] = NULL;
      // 5. t 指向 NULL.
      this->deque_tail[deque_no] = NULL;
    }
  }
  // gc 沒有任何回收的 node.
  else
  {
    // deque 除了這一個要 pop 的 node 還有其他 node 存在.
    if (*(void**)(this->deque_tail[deque_no] + this->data_size) != NULL)
    {
      // 1. gc 指向 t.
      this->gc_ptr = this->deque_tail[deque_no];
      // 2. t 指向 t->p.
      this->deque_tail[deque_no] = (char*)*(void**)(this->deque_tail[deque_no] + this->data_size);
      // 3. gc->p 指向 NULL.
      *(void**)(this->gc_ptr + this->data_size) = NULL;
      // 4. t->n 指向 NULL.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = NULL;
    }
    // deque 只剩這一個要 pop 的 node 存在.
    else
    {
      // gc 指標指向要 pop 的 node. (1. gc 指向 t)
      this->gc_ptr = this->deque_tail[deque_no];
      // deque 的 head 指向 NULL. (2. h 指向 NULL)
      this->deque_head[deque_no] = NULL;
      // deque 的 tail 指向 NULL. (3. t 指向 NULL)
      this->deque_tail[deque_no] = NULL;
    }
  }

  // deque 的 size 減少 1.
  --this->deque_size[deque_no];

  // 將 deque_curr_pos 及 deque_current 清空.
  this->deque_current[deque_no] = NULL;
  this->deque_curr_pos[deque_no] = 0;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::pop_front(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // deque 裡面沒有資料.
  if (this->deque_head[deque_no] == NULL)
  {
    return 1;
  }

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // deque 除了這一個要 pop 的 node 還有其他 node 存在.
    if (*(void**)(this->deque_head[deque_no] + this->data_size + gc_sizeof_ptr) != NULL)
    {
      // gc 的第一個 node 的指向 previous node 的指標指向要 pop 的 node. (1. gc->p 指向 h)
      *(void**)(this->gc_ptr + this->data_size) = this->deque_head[deque_no];
      // deque 的 head 指向下一個 node. (2. h 指向 h->n)
      this->deque_head[deque_no] = (char*)*(void**)(this->deque_head[deque_no] + this->data_size + gc_sizeof_ptr);
      // deque 的 head 的上一個 node 指向 next node 的指標指向 gc 的第一個 node. (3 h->p->n 指向 gc)
      *(void**)((char*)(*(void**)(this->deque_head[deque_no] + this->data_size)) + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // deque 的 head 指向 previous node 的指標指向 NULL. (4. h->p 指向 NULL)
      *(void**)(this->deque_head[deque_no] + this->data_size) = NULL;
      // gc 指標指向自己的上一個 node. (5. gc 指向 gc->p)
      this->gc_ptr = (char*)*(void**)(this->gc_ptr + this->data_size);
    }
    // deque 只剩這一個要 pop 的 node 存在.
    else
    {
      // gc 的第一個 node 的指向 previous node 的指標指向要 pop 的 node. (1. gc->p 指向 h)
      *(void**)(this->gc_ptr + this->data_size) = this->deque_head[deque_no];
      // deque 的 head 指向 next node 的指標指向 gc. (2. h->n 指向 gc)
      *(void**)(this->deque_head[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // gc 指標指向要 pop 的 node. (3. gc 指向 h)
      this->gc_ptr = this->deque_head[deque_no];
      // deque 的 head 指向 NULL. (4. h 指向 NULL)
      this->deque_head[deque_no] = NULL;
      // deque 的 tail 指向 NULL. (5. t 指向 NULL)
      this->deque_tail[deque_no] = NULL;
    }
  }
  // gc 沒有任何回收的 node.
  else
  {
    // deque 除了這一個要 pop 的 node 還有其他 node 存在.
    if (*(void**)(this->deque_head[deque_no] + this->data_size + gc_sizeof_ptr) != NULL)
    {
      // gc 指標指向要 pop 的 node. (1. gc 指向 h)
      this->gc_ptr = this->deque_head[deque_no];
      // deque 的 head 指向下一個 node. (2. h 指向 h->n)
      this->deque_head[deque_no] = (char*)*(void**)(this->deque_head[deque_no] + this->data_size + gc_sizeof_ptr);
      // deque 的 head 的上一個 node 的指向 next node 的指標指向 NULL. (3 h->p->n 指向 NULL)
      *(void**)((char*)(*(void**)(this->deque_head[deque_no] + this->data_size)) + this->data_size + gc_sizeof_ptr) = NULL;
      // deque 的 head 指向 previous node 的指標指向 NULL. (4. h->p 指向 NULL)
      *(void**)(this->deque_head[deque_no] + this->data_size) = NULL;
    }
    // deque 只剩這一個要 pop 的 node 存在.
    else
    {
      // gc 指標指向要 pop 的 node. (1. gc 指向 h)
      this->gc_ptr = this->deque_head[deque_no];
      // deque 的 head 指向 NULL. (2. h 指向 NULL)
      this->deque_head[deque_no] = NULL;
      // deque 的 tail 指向 NULL. (3. t 指向 NULL)
      this->deque_tail[deque_no] = NULL;
    }
  }

  // deque 的 size 減少 1.
  --this->deque_size[deque_no];

  // 將 deque_curr_pos 及 deque_current 清空.
  this->deque_current[deque_no] = NULL;
  this->deque_curr_pos[deque_no] = 0;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::pop_head_to_current(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // current 指標指向 NULL.
  if (this->deque_current[deque_no] == NULL)
  {
    return -2;
  }

  // deque 裡面沒有資料. (如果 deque 沒資料, current 指標一定指向 NULL.)
  //if (this->deque_head[deque_no] == NULL)
  //{
    //return 1;
  //}

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // deque 除了這一個要 pop 的 node(s) 還有其他 node 存在. (current->next 不為 NULL)
    if (*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr) != NULL)
    {
      // 1. gc->prev 指向 current.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_current[deque_no];
      // 2. current 指向 current->next.
      this->deque_current[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr);
      // 3. current->prev->next 指向 gc.
      *(void**)((char*)(*(void**)(this->deque_current[deque_no] + this->data_size)) + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 4. current->prev 指向 NULL.
      *(void**)(this->deque_current[deque_no] + this->data_size) = NULL;
      // 5. gc 指向 head.
      this->gc_ptr = this->deque_head[deque_no];
      // 6. head 指向 current.
      this->deque_head[deque_no] = this->deque_current[deque_no];
      // 7. current 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // 計算 pop 後 deque 的 node 數量.
      this->deque_size[deque_no] -= this->deque_curr_pos[deque_no];
    }
    // deque 只剩要 pop 的 node(s) 存在.
    else
    {
      // 1. gc->prev 指向 current.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_current[deque_no];
      // 2. current->next 指向 gc.
      *(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 3. gc 指向 head.
      this->gc_ptr = this->deque_head[deque_no];
      // 4. head 指向 NULL.
      this->deque_head[deque_no] = NULL;
      // 5. tail 指向 NULL.
      this->deque_tail[deque_no] = NULL;
      // 6. current 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // pop 後 deque 的 node 數量等於 0.
      this->deque_size[deque_no] = 0;
    }
  }
  // gc 沒有任何回收的 node.
  else
  {
    // deque 除了這一個要 pop 的 node(s) 還有其他 node 存在. (current->next 不為 NULL)
    if (*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr) != NULL)
    {
      // 1. gc 指向 head.
      this->gc_ptr = this->deque_head[deque_no];
      // 2. head 指向 current->next.
      this->deque_head[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr);
      // 3. head->prev 指向 NULL.
      *(void**)(this->deque_head[deque_no] + this->data_size) = NULL;
      // 4. current->next 指向 NULL.
      *(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr) = NULL;
      // 5. current 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // 計算 pop 後 deque 的 node 數量.
      this->deque_size[deque_no] -= this->deque_curr_pos[deque_no];
    }
    // deque 只剩要 pop 的 node(s) 存在.
    else
    {
      // 1. gc 指向 head.
      this->gc_ptr = this->deque_head[deque_no];
      // 2. head 指向 NULL.
      this->deque_head[deque_no] = NULL;
      // 3. tail 指向 NULL.
      this->deque_tail[deque_no] = NULL;
      // 4. current 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // pop 後 deque 的 node 數量等於 0.
      this->deque_size[deque_no] = 0;
    }
  }

  // current 指標沒有指到任何 node.
  this->deque_curr_pos[deque_no] = 0;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::pop_head_to_prev(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // current 指標指向 NULL.
  if (this->deque_current[deque_no] == NULL)
  {
    return -2;
  }

  // deque 裡面沒有資料. (如果 deque 沒資料, current 指標一定指向 NULL.)
  //if (this->deque_head[deque_no] == NULL)
  //{
    //return 1;
  //}

  // current->prev 指標指向 NULL, 表示 current 指向 head. 沒有 head 到 current->prev 的資料可以去掉.
  //if (*(void**)(this->deque_current[deque_no] + this->data_size) == NULL)
  if (this->deque_current[deque_no] == this->deque_head[deque_no])
  {
    return 2;
  }

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // 1. gc->prev 指向 current->prev.
    *(void**)(this->gc_ptr + this->data_size) = (char*)*(void**)(this->deque_current[deque_no] + this->data_size);
    // 2. current->prev->next 指向 gc.
    *(void**)((char*)(*(void**)(this->deque_current[deque_no] + this->data_size)) + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
    // 3. current->prev 指向 NULL.
    *(void**)(this->deque_current[deque_no] + this->data_size) = NULL;
    // 4. gc 指向 head.
    this->gc_ptr = this->deque_head[deque_no];
    // 5. head 指向 current.
    this->deque_head[deque_no] = this->deque_current[deque_no];
  }
  // gc 沒有任何回收的 node.
  else
  {
    // 1. gc 指向 head.
    this->gc_ptr = this->deque_head[deque_no];
    // 2. head 指向 current.
    this->deque_head[deque_no] = this->deque_current[deque_no];
    // 3. current->prev->next 指向 NULL.
    *(void**)((char*)(*(void**)(this->deque_current[deque_no] + this->data_size)) + this->data_size + gc_sizeof_ptr) = NULL;
    // 4. current->prev 指向 NULL.
    *(void**)(this->deque_current[deque_no] + this->data_size) = NULL;
  }

  // 計算 pop 後 deque 的 node 數量.
  this->deque_size[deque_no] -= (this->deque_curr_pos[deque_no] - 1);
  // current 指標在 head.
  this->deque_curr_pos[deque_no] = 1;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::pop_current_to_tail(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // current 指標指向 NULL.
  if (this->deque_current[deque_no] == NULL)
  {
    return -2;
  }

  // deque 裡面沒有資料. (如果 deque 沒資料, current 指標一定指向 NULL.)
  //if (this->deque_head[deque_no] == NULL)
  //{
    //return 1;
  //}

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // deque 除了這一個要 pop 的 node(s) 還有其他 node 存在. (current->prev 不為 NULL)
    if (*(void**)(this->deque_current[deque_no] + this->data_size) != NULL)
    {
      // 1. gc->p 指向 t.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_tail[deque_no];
      // 2. t->n 指向 gc.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 3. gc 指向 c.
      this->gc_ptr = this->deque_current[deque_no];
      // 4. t 指向 c->p.
      this->deque_tail[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size);
      // 5. t->n 指向 NULL.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = NULL;
      // 6. c->p 指向 NULL.
      *(void**)(this->deque_current[deque_no] + this->data_size) = NULL;
      // 7. c 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // 計算 pop 後 deque 的 node 數量.
      this->deque_size[deque_no] = this->deque_curr_pos[deque_no] - 1;
    }
    // deque 只剩要 pop 的 node(s) 存在.
    else
    {
      // 1. gc->p 指向 t.
      *(void**)(this->gc_ptr + this->data_size) = this->deque_tail[deque_no];
      // 2. t->n 指向 gc.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
      // 3. gc 指向 c.
      this->gc_ptr = this->deque_current[deque_no];
      // 4. h 指向 NULL.
      this->deque_head[deque_no] = NULL;
      // 5. t 指向 NULL.
      this->deque_tail[deque_no] = NULL;
      // 6. c 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // pop 後 deque 的 node 數量等於 0.
      this->deque_size[deque_no] = 0;
    }
  }
  // gc 沒有任何回收的 node.
  else
  {
    // deque 除了這一個要 pop 的 node(s) 還有其他 node 存在. (current->prev 不為 NULL)
    if (*(void**)(this->deque_current[deque_no] + this->data_size) != NULL)
    {
      // 1. gc 指向 c.
      this->gc_ptr = this->deque_current[deque_no];
      // 2. t 指向 c->p.
      this->deque_tail[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size);
      // 3. t->n 指向 NULL.
      *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = NULL;
      // 4. c->p 指向 NULL.
      *(void**)(this->deque_current[deque_no] + this->data_size) = NULL;
      // 5. c 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // 計算 pop 後 deque 的 node 數量.
      this->deque_size[deque_no] = this->deque_curr_pos[deque_no] - 1;
    }
    // deque 只剩要 pop 的 node(s) 存在.
    else
    {
      // 1. gc 指向 current.
      this->gc_ptr = this->deque_current[deque_no];
      // 2. head 指向 NULL.
      this->deque_head[deque_no] = NULL;
      // 3. tail 指向 NULL.
      this->deque_tail[deque_no] = NULL;
      // 4. current 指向 NULL.
      this->deque_current[deque_no] = NULL;

      // pop 後 deque 的 node 數量等於 0.
      this->deque_size[deque_no] = 0;
    }
  }

  // current 指標沒有指到任何 node.
  this->deque_curr_pos[deque_no] = 0;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::pop_all(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // deque 裡面沒有資料.
  if (this->deque_tail[deque_no] == NULL)
  {
    return 0;
  }

  // gc 已經有回收的 node.
  if (this->gc_ptr != NULL)
  {
    // 1. gc->p 指向 t.
    *(void**)(this->gc_ptr + this->data_size) = this->deque_tail[deque_no];
    // 2. t->n 指向 gc.
    *(void**)(this->deque_tail[deque_no] + this->data_size + gc_sizeof_ptr) = this->gc_ptr;
    // 3. gc 指向 h.
    this->gc_ptr = this->deque_head[deque_no];
    // 4. h 指向 NULL.
    this->deque_head[deque_no] = NULL;
    // 5. t 指向 NULL.
    this->deque_tail[deque_no] = NULL;
    // 6. c 指向 NULL.
    this->deque_current[deque_no] = NULL;

    // pop 後 deque 的 node 數量等於 0.
    this->deque_size[deque_no] = 0;
  }
  // gc 沒有任何回收的 node.
  else
  {
    // 1. gc 指向 head.
    this->gc_ptr = this->deque_head[deque_no];
    // 2. head 指向 NULL.
    this->deque_head[deque_no] = NULL;
    // 3. tail 指向 NULL.
    this->deque_tail[deque_no] = NULL;
    // 4. current 指向 NULL.
    this->deque_current[deque_no] = NULL;

    // pop 後 deque 的 node 數量等於 0.
    this->deque_size[deque_no] = 0;
  }

  // current 指標沒有指到任何 node.
  this->deque_curr_pos[deque_no] = 0;

  // 將 deque_at_pos 及 deque_at 清空.
  this->deque_at[deque_no] = NULL;
  this->deque_at_pos[deque_no] = 0;

  return 0;
}

int16_t mws_fast_deque::start_back(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // deque 裡面沒有資料.
  if (this->deque_tail[deque_no] == NULL)
  {
    // current 指標指向 NULL.
    this->deque_current[deque_no] = NULL;
    // current 指標沒有指到任何 node.
    this->deque_curr_pos[deque_no] = 0;
    return 1;
  }

  // 將 current 指標指向 tail 指標指向的 node.
  this->deque_current[deque_no] = this->deque_tail[deque_no];
  // current 指標指到最後一個 node, 位置等於 size.
  this->deque_curr_pos[deque_no] = this->deque_size[deque_no];

  return 0;
}

int16_t mws_fast_deque::start_front(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // deque 裡面沒有資料.
  if (this->deque_head[deque_no] == NULL)
  {
    // current 指標指向 NULL.
    this->deque_current[deque_no] = NULL;
    // current 指標沒有指到任何 node.
    this->deque_curr_pos[deque_no] = 0;
    return 1;
  }

  // 將 current 指標指向 head 指標指向的 node.
  this->deque_current[deque_no] = this->deque_head[deque_no];
  // current 指標指到第一個 node, 位置等於 1.
  this->deque_curr_pos[deque_no] = 1;

  return 0;
}

int16_t mws_fast_deque::read_prev(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // current 指標指向 NULL.
  if (this->deque_current[deque_no] == NULL)
  {
    return -2;
  }

  // current 的 previous node 的指標指向 NULL, 往前已經沒有 node 了.
  if ((char*)*(void**)(this->deque_current[deque_no] + this->data_size) == NULL)
  {
    return 1;
  }

  // current 指向 previous node.
  this->deque_current[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size);
  // current 指標指向的位置減 1.
  --this->deque_curr_pos[deque_no];

  return 0;
}

int16_t mws_fast_deque::read_next(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // current 指標指向 NULL.
  if (this->deque_current[deque_no] == NULL)
  {
    return -2;
  }

  // current 的 next node 的指標指向 NULL, 往後已經沒有 node 了.
  if ((char*)*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr) == NULL)
  {
    return 1;
  }

  // current 指向 next node.
  this->deque_current[deque_no] = (char*)*(void**)(this->deque_current[deque_no] + this->data_size + gc_sizeof_ptr);
  // current 指標指向的位置加 1.
  ++this->deque_curr_pos[deque_no];

  return 0;
}

int16_t mws_fast_deque::reset_current(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  // reset current 指向 NULL.
  this->deque_current[deque_no] = NULL;
  // reset current 指標指向的位置為 0.
  this->deque_curr_pos[deque_no] = 0;

  return 0;
}

void* mws_fast_deque::at(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return NULL;
  }

  return this->deque_current[deque_no];
}

void* mws_fast_deque::at(ssize_t deque_no, ssize_t pos)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return NULL;
  }

  // pos 超出範圍, 不存在這個 node.
  if ((pos >= this->deque_size[deque_no]) ||
      (pos < 0))
  {
    return NULL;
  }

  // 內部運算時, 將 pos 加 1.
  pos += 1;

  // 要指向要查詢的 node 的指標.
  char* ptr = NULL;

  // pos 在上次的 next.
  if ((this->deque_at_pos[deque_no] > 0) &&
      (pos == (this->deque_at_pos[deque_no] + 1)))
  {
    ptr = (char*)*(void**)(this->deque_at[deque_no] + this->data_size + gc_sizeof_ptr);
    //printf("at test: case 1\n");
  }
  // pos 在上次的 prev.
  else if (pos == (this->deque_at_pos[deque_no] - 1))
  {
    ptr = (char*)*(void**)(this->deque_at[deque_no] + this->data_size);
    //printf("at test: case 2\n");
  }
  // pos 和上次相同.
  else if (pos == this->deque_at_pos[deque_no])
  {
    ptr = this->deque_at[deque_no];
    //printf("at test: case 3\n");
  }
  // pos 和 curr_pos 相同.
  else if (pos == this->deque_curr_pos[deque_no])
  {
    ptr = this->deque_current[deque_no];
    //printf("at test: case 4\n");
  }
  else
  {
    ssize_t offset_head_to_pos = pos - 1;
    ssize_t offset_pos_to_tail = this->deque_size[deque_no] - pos;
    ssize_t offset_curr_to_pos = pos - this->deque_curr_pos[deque_no];
    ssize_t offset_last_to_pos = pos - this->deque_at_pos[deque_no];
    // 從 head 開始找.
    if ((offset_head_to_pos <= offset_pos_to_tail) &&
        (offset_head_to_pos <= abs((int)offset_curr_to_pos)) &&
        (offset_head_to_pos <= abs((int)offset_last_to_pos)))
    {
      ptr = this->deque_head[deque_no];
      // 往 ptr 的 next 方向走.
      for (ssize_t i = 0; i < offset_head_to_pos; ++i)
      {
        ptr = (char*)*(void**)(ptr + this->data_size + gc_sizeof_ptr);
      }
      //printf("at test: case 5\n");
    }
    // 從 tail 開始找.
    else if ((offset_pos_to_tail <= abs((int)offset_curr_to_pos)) &&
             (offset_pos_to_tail <= abs((int)offset_last_to_pos)))
    {
      ptr = this->deque_tail[deque_no];
      // 往 ptr 的 prev 方向走.
      for (ssize_t i = 0; i < offset_pos_to_tail; ++i)
      {
        ptr = (char*)*(void**)(ptr + this->data_size);
      }
      //printf("at test: case 6\n");
    }
    // 從 current 開始找.
    else if (abs((int)offset_curr_to_pos) <= abs((int)offset_last_to_pos))
    {
      ptr = this->deque_current[deque_no];
      // current 在 pos 前面.
      if (offset_curr_to_pos >= 0)
      {
        // 往 ptr 的 next 方向走.
        for (ssize_t i = 0; i < offset_curr_to_pos; ++i)
        {
          ptr = (char*)*(void**)(ptr + this->data_size + gc_sizeof_ptr);
        }
        //printf("at test: case 7\n");
      }
      // current 在 pos 後面.
      else
      {
        // 往 ptr 的 prev 方向走.
        for (ssize_t i = offset_curr_to_pos; i < 0; ++i)
        {
          ptr = (char*)*(void**)(ptr + this->data_size);
        }
        //printf("at test: case 8\n");
      }
    }
    // 從 last 開始找.
    else
    {
      ptr = this->deque_at[deque_no];
      // last 在 pos 前面.
      if (offset_last_to_pos >= 0)
      {
        // 往 ptr 的 next 方向走.
        for (ssize_t i = 0; i < offset_last_to_pos; ++i)
        {
          ptr = (char*)*(void**)(ptr + this->data_size + gc_sizeof_ptr);
        }
        //printf("at test: case 9\n");
      }
      // last 在 pos 後面.
      else
      {
        // 往 ptr 的 prev 方向走.
        for (ssize_t i = offset_last_to_pos; i < 0; ++i)
        {
          ptr = (char*)*(void**)(ptr + this->data_size);
        }
        //printf("at test: case 10\n");
      }
    }
  }

  // 更新 deque_at.
  this->deque_at[deque_no] = ptr;
  // 更新 deque_at_pos.
  this->deque_at_pos[deque_no] = pos;

  return ptr;
}

ssize_t mws_fast_deque::size(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  return this->deque_size[deque_no];
}

bool mws_fast_deque::empty(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return true;
  }

  // 沒有任何 node 在這個 deque.
  if (this->deque_size[deque_no] == 0)
  {
    return true;
  }

  return false;
}

ssize_t mws_fast_deque::curr_pos(ssize_t deque_no)
{
  // 不存在這個 deque.
  if (deque_no >= this->next_deque_no)
  {
    return -1;
  }

  if (this->deque_curr_pos[deque_no] == 0)
  {
    return -2;
  }

  return (this->deque_curr_pos[deque_no] - 1);
}

int mws_fast_deque::lock()
{
  return pthread_mutex_lock(&(this->mutex));
}

int mws_fast_deque::trylock()
{
  return pthread_mutex_trylock(&(this->mutex));
}

int mws_fast_deque::unlock()
{
  return pthread_mutex_unlock(&(this->mutex));
}

void mws_fast_deque::show_deque_content(ssize_t deque_no)
{
  printf("========== dq# %d content, size: %d ==========\n", (int)deque_no, (int)this->deque_size[deque_no]);

  char* current_ptr = this->deque_head[deque_no];

  //printf("ln:%d  current_ptr = %p\n", __LINE__, current_ptr);

  int32_t note_no = 0;
  while (current_ptr != NULL)
  {
    //std::cout << __FILE__ << " : " << __LINE__ << std::endl;

    printf("addr: %p, node %d: ", current_ptr, ++note_no);
    for (ssize_t i = 0;
         i < (this->data_size + gc_sizeof_2_ptr);
         ++i)
    {
      printf("%02x ", *(uint8_t*)((char*)current_ptr + i));
    }
    printf("\n");

    //void* ptr = current_ptr + this->data_size + sizeof(void*);
    //printf("  current_ptr = %p\n", current_ptr);
    //printf("  ptr = %p\n", ptr);
    //printf("  *(void**)ptr = %p\n", *(void**)ptr);

    //current_ptr = (char*)*(void**)ptr;
    current_ptr = (char*)*(void**)(current_ptr + this->data_size + gc_sizeof_ptr);
  }

  printf("deque_current: %p, deque_curr_pos: %d\n",
         this->deque_current[deque_no],
         (int)this->deque_curr_pos[deque_no]);

  printf("deque_at: %p, deque_at_pos: %d\n",
         this->deque_at[deque_no],
         (int)this->deque_at_pos[deque_no]);

  printf("================================\n");

  return;
}

void mws_fast_deque::show_gc_content()
{
  printf("========== gc content ==========\n");

  char* current_ptr = this->gc_ptr;

  //printf("ln:%d  current_ptr = %p\n", __LINE__, current_ptr);

  int32_t note_no = 0;
  while (current_ptr != NULL)
  {
    //std::cout << __FILE__ << " : " << __LINE__ << std::endl;
    printf("gc addr: %p, node %d: ", current_ptr, ++note_no);
    for (ssize_t i = 0;
         i < (this->data_size + gc_sizeof_2_ptr);
         ++i)
    {
      printf("%02x ", *(uint8_t*)((char*)current_ptr + i));
    }
    printf("\n");

    //void* ptr = current_ptr + this->data_size + sizeof(void*);
    //printf("  current_ptr = %p\n", current_ptr);
    //printf("  ptr = %p\n", ptr);
    //printf("  *(void**)ptr = %p\n", *(void**)ptr);

    //current_ptr = (char*)*(void**)ptr;
    current_ptr = (char*)*(void**)(current_ptr + this->data_size + gc_sizeof_ptr);
  }

  printf("================================\n");

  return;
}

char* mws_fast_deque::get_a_node()
{
  char* node = NULL;
  if (this->gc_ptr != NULL)
  {
    // 取 gc 的第一個 node.
    node = this->gc_ptr;
    // 將 gc 指標指向下一個 node.
    this->gc_ptr = (char*)*(void**)(this->gc_ptr + this->data_size + gc_sizeof_ptr);

    // 維護 node 指向 previous node 的指標: 指向 NULL. (可以不維護)
    //*(void**)(node + this->data_size) = NULL;
    // 維護 node 指向 next node 的指標: 指向 NULL. (可以不維護)
    //*(void**)(node + this->data_size + gc_sizeof_ptr) = NULL;

    // 維護 gc 第一個 node 指向 previous node 的指標: 指向 NULL. (可以不維護)
    //*(void**)(this->gc_ptr + this->data_size) = NULL;
  }
  else
  {
    // 如果 pool 中可用的 node 數量為 0, 則需要在 pool 中加一個新的 block.
    if (this->number_of_nodes_in_pool <= 0)
    {
      // 向作業系統要一塊新記憶體空間給 pool.
      this->pool_ptr = (char*)calloc(this->number_of_node_in_new_block, (data_size + gc_sizeof_2_ptr));
      // 記錄 block 的起始位置.
      this->block.push_back(this->pool_ptr);
      // 更新在 pool 中可用的 node 數量.
      this->number_of_nodes_in_pool = this->number_of_node_in_new_block;
    }

    // 取 pool 的第一個 node.
    node = this->pool_ptr;

    // 在 pool 中可用的 node 數量減 1.
    --this->number_of_nodes_in_pool;

    // 如果 pool 還有下一個 node.
    if (this->number_of_nodes_in_pool > 0)
    {
      // 將 pool 指標指向下一個 node.
      this->pool_ptr = this->pool_ptr + this->data_size + gc_sizeof_2_ptr;
    }
    // 如果 pool 中已經沒有可用的 node.
    else
    {
      // 將 pool 指標指向 NULL.
      this->pool_ptr = NULL;
    }
    // 維護 node 指向 previous node 的指標: 指向 NULL. (可以不維護)
    //*(void**)(node + this->data_size) = NULL;
    // 維護 node 指向 next node 的指標: 指向 NULL. (可以不維護)
    //*(void**)(node + this->data_size + gc_sizeof_ptr) = NULL;
  }

  //char* node = (char*)calloc(1, (data_size + gc_sizeof_2_ptr));

  return node;
}
