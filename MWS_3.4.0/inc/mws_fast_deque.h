// 日期      版本       維護人員    修改原因.
// 20200206  v01.00.00  吳青華      新程式開發.

#ifndef MWS_FAST_DEQUE_H_INCLUDED
#define MWS_FAST_DEQUE_H_INCLUDED

#include <sys/types.h>

#include <pthread.h>
#include <stdint.h>

#include <cstring>
#include <vector>

#define DEFAULT_NODE_QUANTITY 10000
#define EXTENDED_NODE_QUANTITY 1000

class mws_fast_deque;

typedef mws_fast_deque mws_fast_deque_t;

class mws_fast_deque
{
  public:
    // 功能: 建構式(constructor), 用以初始化物件.
    // 回傳值: 無.
    // 參數:
    //    data_size: 資料長度.
    //    default_node_qty: 預計此物件需要的 node 數.
    //    extended_node_qty: 需要額外新空間時, 新空間可容納的 node 數.
    mws_fast_deque(ssize_t data_size,
                   ssize_t default_node_qty = DEFAULT_NODE_QUANTITY,
                   ssize_t extended_node_qty = EXTENDED_NODE_QUANTITY);
    // 功能: 解構式(destructor).
    // 回傳值: 無.
    // 參數: 無.
    ~mws_fast_deque();
    // 功能: 取得資料長度.
    // 回傳值: 資料長度.
    // 參數: 無.
    ssize_t get_data_size();
    // 功能: 取得一個新的 deque.
    // 回傳值: 新的 deque 的代號.
    // 參數: 無.
    ssize_t get_new_deque();
    // 功能: 選擇一個 deque, 並新增一個 node 在最後面.
    // 回傳值:
    //   新增的 node 的指標.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    char* push_back_a_black_block(ssize_t deque_no);
    // 功能: 選擇一個 deque, 並新增一筆資料在最後面.
    // 回傳值:
    //    0: 新增資料完成.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    //   data_ptr: 指向要新增的資料的指標.
    int16_t push_back(ssize_t deque_no, void* data_ptr);
    // 功能: 選擇一個 deque, 並新增一筆資料在最前面.
    // 回傳值:
    //    0: 新增資料完成.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    //   data_ptr: 指向要新增的資料的指標.
    int16_t push_front(ssize_t deque_no, void* data_ptr);
    // 功能: 選擇一個 deque, 並從最後面去掉一筆資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //    1: deque 裡面沒有資料可以去掉.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_back(ssize_t deque_no);
    // 功能: 選擇一個 deque, 並從最前面去掉一筆資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //    1: deque 裡面沒有資料可以去掉.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_front(ssize_t deque_no);
    // 功能: 選擇一個 deque, 並從最前面(head)到 current 指標所在的位置一次去掉多筆資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //XXX 1: deque 裡面沒有資料可以去掉.
    //   -1: 不存在這個 deque.
    //   -2: current 指標指向 NULL.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_head_to_current(ssize_t deque_no);
    // 功能: 選擇一個 deque, 並從最前面(head)到 current 的 prev 指標所在的位置一次去掉多筆資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //XXX 1: deque 裡面沒有資料可以去掉.
    //    2: 沒有 head 到 current->prev 的資料可以去掉.
    //   -1: 不存在這個 deque.
    //   -2: current 指標指向 NULL.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_head_to_prev(ssize_t deque_no);
    // 功能: 選擇一個 deque, 並從 current 指標到最後面(tail)所在的位置一次去掉多筆資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //XXX 1: deque 裡面沒有資料可以去掉.
    //   -1: 不存在這個 deque.
    //   -2: current 指標指向 NULL.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_current_to_tail(ssize_t deque_no);
    // 功能: 選擇一個 deque, 清除掉該 deque 的所有資料.
    // 回傳值:
    //    0: 去掉資料完成.
    //XXX 1: deque 裡面沒有資料可以去掉.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t pop_all(ssize_t deque_no);
    // 功能: 選擇一個 deque, 將 current 指標指向 tail 指標指向的 node.
    // 回傳值:
    //    0: start 完成.
    //    1: deque 裡面沒有資料, current 指向 NULL.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t start_back(ssize_t deque_no);
    // 功能: 選擇一個 deque, 將 current 指標指向 head 指標指向的 node.
    // 回傳值:
    //    0: start 完成.
    //    1: deque 裡面沒有資料, current 指向 NULL.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t start_front(ssize_t deque_no);
    // 功能:  選擇一個 deque, 將 current 指標指向原來所指的 node 的 previous node.
    // 回傳值:
    //    0: 移動完成.
    //    1: 不存在 previous node, current 指標不動.
    //   -1: 不存在這個 deque.
    //   -2: current 指標指向 NULL..
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t read_prev(ssize_t deque_no);
    // 功能:  選擇一個 deque, 將 current 指標指向原來所指的 node 的 next node.
    // 回傳值:
    //    0: 移動完成.
    //    1: 不存在 next node, current 指標不動.
    //   -1: 不存在這個 deque.
    //   -2: current 指標指向 NULL..
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t read_next(ssize_t deque_no);
    // 功能:  選擇一個 deque, reset current 指標(之前的start、read_prev、read_next無效).
    // 回傳值:
    //    0: reset完成.
    //   -1: 不存在這個 deque.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    int16_t reset_current(ssize_t deque_no);
    // 功能:  選擇一個 deque, 取得 current 指標指向的 node 的資料.
    // 回傳值: current 指標指向的 node 的資料.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    void* at(ssize_t deque_no);
    // 功能:  選擇一個 deque, 取得指定的 node 的資料.
    // 回傳值: current 指標指向的 node 的資料.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    //   pos: 指定位的 node 的位置(第一個 node 位置為 0).
    void* at(ssize_t deque_no, ssize_t pos);
    // 功能:  選擇一個 deque, 取得該 deque 的 node 數量.
    // 回傳值:
    //   -1: 不存在這個 deque.
    //   >= 0: node 的數量.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    ssize_t size(ssize_t deque_no);
    // 功能:  選擇一個 deque, 檢查該 deque 是否有 node.
    // 回傳值:
    //   true: 沒有 node, 包括不存在這個 deque.
    //   false: 有 node.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    bool empty(ssize_t deque_no);
    // 功能:  選擇一個 deque, 取得該 deque 的 current 指標所在的 node 位置.
    // 回傳值:
    //   -1: 不存在這個 deque.
    //   -2: current 指標尚未指向任何 node 位置.
    //   >= 0: current 指標所在的 node 位置(第一個 node 位置為 0).
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    ssize_t curr_pos(ssize_t deque_no);

    // 功能: 需要達到 thread-safe 時用來 lock 的函式.
    // 回傳值:
    //   0: lock 成功.
    //   非0: lock 失敗.
    // 參數: 無.
    int lock();
    // 功能: 需要達到 thread-safe 時用來 try lock 的函式(非阻塞).
    // 回傳值:
    //   0: lock 成功.
    //   非0: lock 失敗.
    //     EBUSY: 失敗原因, 已經被 lock 了.
    // 參數: 無.
    int trylock();
    // 功能: 需要達到 thread-safe 時用來 unlock 的函式.
    // 回傳值:
    //   0: unlock 成功.
    //   非0: unlock 失敗.
    // 參數: 無.
    int unlock();

    // 功能: debug 用, 將所選擇的 deque 的所有 node 的資料顯示在螢幕.
    // 回傳值: 無.
    // 參數:
    //   deque_no: 所選擇的 deque 的代號.
    void show_deque_content(ssize_t deque_no);
    // 功能: debug 用, 將 gc 的所有 node 的資料顯示在螢幕.
    // 回傳值: 無.
    // 參數: 無.
    void show_gc_content();

  private:
    // 資料長度.
    ssize_t data_size;
    // pointer of the pool of available node block.
    char* pool_ptr;
    // number of remained available node in the pool.
    ssize_t number_of_nodes_in_pool;
    // number of node in new block for pool.
    ssize_t number_of_node_in_new_block;
    // pointer of garbage collector.
    char* gc_ptr;
    // 下一個新的 deque 的代號.
    ssize_t next_deque_no;
    // 所有的 deque 的 head.
    std::vector<char*> deque_head;
    // 所有的 deque 的 tail.
    std::vector<char*> deque_tail;
    // 所有的 deque 的 current.
    std::vector<char*> deque_current;
    // 所有的 deque 的 current 指標所在的位置 (第一個 node 位置為 1).
    std::vector<ssize_t> deque_curr_pos;
    // 所有的 deque 的 void* at(ssize_t deque_no, ssize_t pos) 上次指向的 node 位置(指標).
    std::vector<char*> deque_at;
    // 所有的 deque 的 void* at(ssize_t deque_no, ssize_t pos) 上次指向的位置 (第一個 node 位置為 1).
    std::vector<ssize_t> deque_at_pos;
    // 所有的 deque 的 node 數量.
    std::vector<ssize_t> deque_size;
    // 記錄所有向作業系統要來的 block 的起始位置.
    std::vector<char*> block;

    // 用來做 lock 和 unlock.
    pthread_mutex_t mutex;

    char* get_a_node();
};

#endif // MWS_FAST_DEQUE_H_INCLUDED
