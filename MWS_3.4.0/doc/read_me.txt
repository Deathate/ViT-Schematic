＊ 函式庫使用方式：
  － 只需要 include mws.h。
  － 一開始需要先呼叫函式
     int mws_init(const std::string program_name,
                  const std::string mws_cfg_file_path,
                  const std::string mws_log_file_path);
     用以初始化 mws。

＊ MWS configuration 說明：
  － 關於 evq 設定值：
     1. 所有 evq 有一個共同的預設 section：evq_default。
     2. 參數 is_auto_dispatch：是否自動 dispatch event，預設為 Y。
        Y 表示呼叫 mws_event_dispatch() 後會自動 dispatch event。
        N 表示要呼叫 mws_event_dispatch() 後，如果已經沒有 event 會自動返回，
        如果需要再 dispatch event 則需要再呼叫一次 mws_event_dispatch()。
  － 關於 ctx 設定值：
     1. 所有 ctx 有一個共同的預設 section：ctx_default。
     2. 參數 pthread_stack_size：pthread 的 size，預設為 0（系統預設）。
        預設值（會依環境改變, 以下參數為 2022/11/23 於測試系統取得）
        - linux: 8388608 bytes。
        - NSK: 131072 bytes。
        PTHREAD_STACK_MIN
        - linux: 16384 bytes。
        - NSK: 4096 bytes（但實際上要大於 32768 bytes,
                           直接使用 PTHREAD_STACK_MIN 會失敗）。
        設定時值只要大於以下值就可以：
        - linux: 16384 bytes。
        - NSK: 32768 bytes。
        NSK 受保護的最小值為 PTHREAD_STACK_MIN_PROTECTED_STACK (49152)。
        NSK 受保護的最大值為 PTHREAD_STACK_MAX_PROTECTED_STACK (16777216)。
        NSK 不受保護的最大值為 PTHREAD_STACK_MAX_NP (33554432)。
  － 關於 src 設定值：
     1. 所有 src 有一個共同的預設 section：src_default。
     2. 參數 topic_name：source 的 topic name，沒有預設值，一定要設定。
     3. 參數 listen_ip：source 使用的 listen IP，預設為 127.0.0.1。
     4. 參數 listen_port：source 使用的 listen port，預設為 1000。
     5. 參數 is_hot_failover_recv_mode：source 是否使用 hot failover mode 接收訊息，預設為 N。
  － 關於 rcv 和 rcv_default：
     1. 所有 rcv 有一個共同的預設 section：rcv_default。
     2. 參數 topic_name：receiver 的 topic name，沒有預設值，一定要設定。
     3. 參數 sess_addr_pair_XX：此 rcv 所負責的所有 session 的 socket (listen & connect) 組合，XX 從01到99，
        source address 預設為 127.0.0.1:1000，receiver address 預設為 127.0.0.1:1-65535。
        - 可以設定多組 session address pair，範圍從 sess_addr_pair_01 到 sess_addr_pair_99。
        - 如果只設定 source address，receiver address 會自動帶入預設值。
     4. 參數 is_hot_failover_recv_mode：receiver 是否使用 hot failover mode 接收訊息，預設為 N。

＊ API 說明：
  － 關於 configuration：
    ------------------------------------------------------------
    // 功能: 初始化 mws
    //       1. 載入 config file.
    //       2. 設定 log file 位置.
    // 回傳值 0: 初始化完成.
    //        1:  已經做過初始化, 本次呼叫無效.
    //        -1: 初始化失敗.
    // 參數 program_name: AP 程式名稱.
    //      mws_cfg_file_path: config file 的位置.
    //      mws_log_file_path: log file 的位置.
    int mws_init(const std::string program_name,
                 const std::string mws_cfg_file_path,
                 const std::string mws_log_file_path);
    ------------------------------------------------------------
    // 功能: 回傳所有的設定值.
    // 回傳值: 所有的設定值.
    // 參數: 無.
    std::map<std::string, std::map<std::string, std::string> > mws_get_cfg();
    ------------------------------------------------------------

  － 關於 evq 及 evq attribute：
    ------------------------------------------------------------
    ● evq attribute type：mws_evq_attr_t
    ● evq attribute 內容：
      1. is_auto_dispatch.
    ● mws_evq_attr_t 變數宣告方式：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式: is_auto_dispatch:
        //                true: auto dispatch.
        //                false: manual dispatch.
        mws_evq_attr_t obj(std::string cfg_section);
    ● mws_evq_attr_t 變數內容修改方式：
      e.g.
        // 回傳值: 沒有.
        // 參數: attr_name: attribute member 的名稱.
        // 參數: attr_value: attribute member 的值.
        obj.mws_modify_evq_attr(std::string attr_name,
                                std::string attr_value);
    ------------------------------------------------------------
    ● evq type：mws_evq_t
    ● mws_evq_t 變數宣告方式（可在程式中宣告 mws_evq_t 變數之前調整屬性）：
      e.g.
        // 參數: mws_evq_attr: event queue attribute 物件.
        mws_evq_t obj(mws_evq_attr_t mws_evq_attr);
    ● mws_evq_t 變數宣告方式（不可在程式中調整屬性）：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式: is_auto_dispatch:
        //                true: auto dispatch.
        //                false: manual dispatch.
        mws_evq_t obj(std::string cfg_section);
    ● mws_evq_t 如何 dispatch 該 event queue 的 event：
      e.g.
        // 回傳值:    0: 正常.
        //         非 0: 異常/失敗 (可由 AP programmer 在 callback function 中定義).
        // 參數: 無.
        int rtv = obj.mws_event_dispatch();
    ● mws_evq_t 如何取得自己的 cfg section name：
      e.g.
        // 回傳值: evq 的 cfg section name.
        // 參數: 無.
        std::string s = obj.mws_get_cfg_section();

  － 關於 ctx 及 ctx attribute：
    ------------------------------------------------------------
    ● ctx attribute type：mws_ctx_attr_t
    ● ctx attribute 內容：
      1. pthread_stack_size.
    ● mws_ctx_attr_t 變數宣告方式：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式:
        //   pthread_stack_size:
        //     the minimum size (in bytes) that will be allocated
        //     for select thread's creation.
        //     pthread_stack_size 的設定：
        //     1. 預設值 (會依環境改變, 以下參數為 2022/11/23 於測試系統取得)
        //       - linux: 8388608 bytes
        //       - NSK: 131072 bytes
        //     2. PTHREAD_STACK_MIN
        //       - linux: 16384 bytes
        //       - NSK: 4096 bytes (但實際上要大於 32768 bytes,
        //                          直接使用 PTHREAD_STACK_MIN 會失敗)
        //     3. 設定時值只要大於以下值就可以：
        //       - linux: 16384 bytes
        //       - NSK: 32768 bytes
        //     4. NSK 受保護的最小值為 PTHREAD_STACK_MIN_PROTECTED_STACK (49152)
        //        NSK 受保護的最大值為 PTHREAD_STACK_MAX_PROTECTED_STACK (16777216)
        //        NSK 不受保護的最大值為 PTHREAD_STACK_MAX_NP (33554432)
        mws_ctx_attr_t obj(std::string cfg_section);
    ● mws_ctx_attr_t 變數內容修改方式：
      e.g.
        // 回傳值: 沒有.
        // 參數: attr_name: attribute member 的名稱.
        // 參數: attr_value: attribute member 的值.
        obj.mws_modify_ctx_attr(std::string attr_name,
                                std::string attr_value);
    ------------------------------------------------------------
    ● ctx 注意事項：宣告一個 ctx 時會啟動一個 thread。
    ● ctx type：mws_ctx_t
    ● mws_ctx_t 變數宣告方式（可在程式中宣告 mws_ctx_t 變數之前調整屬性）：
      e.g.
        // 參數: mws_ctx_attr: context attribute 物件.
        mws_ctx_t obj(mws_ctx_attr_t mws_ctx_attr);
    ● mws_ctx_t 變數宣告方式（不可在程式中調整屬性）：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式:
        //   pthread_stack_size:
        //     the minimum size (in bytes) that will be allocated
        //     for select thread's creation.
        mws_ctx_t obj(std::string cfg_section);
    ● mws_ctx_t 如何取得自己的 cfg section name：
      e.g.
        // 回傳值: ctx 的 cfg section name.
        // 參數: 無.
        std::string s = obj.mws_get_cfg_section();
    ● mws_ctx_t 如何停止 ctx thread：
      e.g.
        // 回傳值: 無.
        // 參數: 無.
        obj.stop_ctx_thread();
    ------------------------------------------------------------

  － 關於 timer callback：
    ------------------------------------------------------------
    ● timer callback 注意事項：
        1. 要依附在一個 ctx 上。
        2. 如果有指定 evq，則由 evq 的 dispatch thread 來執行。
           如果沒有指定 evq，則由 ctx 的 thread 來執行。
    ● timer 排程（microsecond）：
      e.g.
        // 回傳值: >= 0: Timer ID (successful completion).
        //         -2: delay_usec > MAX_DELAY_USEC;
        //         -3: Number of timer >= MAX_TIMER_NUM.
        // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
        //       cb_function: The function to call when the timer expires.
        //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
        //       delay_usec: Delay until cb_function should be called (in microsecond(s)).
        //                   (Max delay_usec is 10,000,000 microseconds (10 seconds)).
        //       is_recurring: Schedule a recurring timer that calls proc when it expires.
        int32_t i = ctx_obj.mws_schedule_timer(mws_evq_t* evq_ptr,
                                               timer_callback_t cb_function,
                                               void* custom_data_ptr,
                                               long delay_usec,
                                               bool is_recurring);
    ● timer 排程（second + microsecond）：
      e.g.
        // 回傳值: >= 0: Timer ID (successful completion).
        //         -1: delay_sec > MAX_DELAY_SEC;
        //         -2: delay_usec > MAX_DELAY_USEC;
        //         -3: Number of timer >= MAX_TIMER_NUM.
        // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
        //       cb_function: The function to call when the timer expires.
        //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
        //       delay_sec: Part of delay time (in second(s)).
        //                  (Max delay_sec is 100,000 seconds).
        //       delay_usec: Part of delay time (in microsecond(s)).
        //                   (Max delay_usec is 10,000,000 microseconds (10 seconds)).
        //       is_recurring: Schedule a recurring timer that calls proc when it expires.
        int32_t i = ctx_obj.mws_schedule_timer(mws_evq_t* evq_ptr,
                                               timer_callback_t cb_function,
                                               void* custom_data_ptr,
                                               long delay_sec,
                                               long delay_usec,
                                               bool is_recurring);
    ● timer 排程（tmvl_t）：
      e.g.
        // 回傳值: >= 0: Timer ID (successful completion).
        //         -3: Number of timer >= MAX_TIMER_NUM.
        // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
        //       cb_function: The function to call when the timer expires.
        //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
        //       time_tv: Exact time in tmvl_t(timeval) format
        int32_t i = mws_schedule_timer(mws_evq_t* evq_ptr,
                                       timer_callback_t cb_function,
                                       void* custom_data_ptr,
                                       tmvl_t time_tv);
    ● timer 排程（年月日時分等）：
      e.g.
        // 回傳值: >= 0: Timer ID (successful completion).
        //         -1: Conversion failed - mktime() failed
        //         -2: year(>= 1900) or
        //             mon(1-12) or
        //             day(1-31) or
        //             hour(0-23) or
        //             min(0-59) or
        //             sec(0-61) or
        //             usec(0-999999)
        //             is out of range.
        //         -3: Number of timer >= MAX_TIMER_NUM.
        // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
        //       cb_function: The function to call when the timer expires.
        //       custom_data_ptr: Pointer to custom data that is passed when the timer expires.
        //       time_tv: Exact time in tmvl_t(timeval) format
        int32_t i = mws_schedule_timer(mws_evq_t* evq_ptr,
                                       timer_callback_t cb_function,
                                       void* custom_data_ptr,
                                       int year,
                                       int mon,
                                       int mday,
                                       int hour,
                                       int min,
                                       int sec,
                                       int usec,
                                       int isdst);
    ● timer 刪除：
      e.g.
        // 回傳值: 0: Timer cancelled.
        //         1: Timer does not exist(or no longer available).
        //         -1: timer_id >= MAX_TIMER_NUM.
        // 參數: evq_ptr: Pointer to event queue with mws_timer_callback_t object.
        //       timer_id: The identifier specifying the timer to cancel.
        int32_t i= mws_cancel_timer(mws_evq_t* evq_ptr,
                                    const int32_t timer_id);
    ------------------------------------------------------------

  － 關於 src 及 src attribute：
    ------------------------------------------------------------
    ● src attribute type：mws_src_attr_t
    ● src attribute 內容：
      1. topic_name.
      2. listen_ip.
      3. listen_port.
      4. is_hot_failover_recv_mode.
    ● mws_src_attr_t 變數宣告方式：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式: topic_name: 關注的 topic name.
        //              listen_ip: 此 src 的 ip.
        //              listen_port: 此 src 的 port.
        //              is_hot_failover_recv_mode:
        //                true: hot failover 接收資料模式,
        //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
        //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
        mws_src_attr_t obj(std::string cfg_section);
    ● mws_src_attr_t 變數內容修改方式：
      e.g.
        // 回傳值: 沒有.
        // 參數: attr_name: attribute member 的名稱.
        // 參數: attr_value: attribute member 的值.
        obj.mws_modify_src_attr(std::string attr_name,
                                std::string attr_value);
    ------------------------------------------------------------
    ● src 注意事項：宣告一個 src 時需要綁定一個 ctx 和 evq。
    ● src type：mws_src_t
    ● mws_src_t 變數宣告方式（可在程式中宣告 mws_src_t 變數之前調整屬性）：
      e.g.
        // 參數: mws_src_attr: source attribute 物件.
        //       ctx_ptr: 屬於哪個 ctx.
        //       evq_ptr: 使用的 evq.
        //       src_cb: event 執行的 callback function.
        //       custom_data_ptr: callback function 的引數 (optional).
        //       custom_data_size: callback function 的引數大小 (byte) (optional).
        mws_src_t obj(mws_src_attr_t mws_src_attr,
                      mws_ctx_t* ctx_ptr,
                      mws_evq_t* evq_ptr,
                      callback_t src_cb,
                      void* custom_data_ptr = NULL,
                      const size_t custom_data_size = 0);
    ● mws_src_t 變數宣告方式（不可在程式中調整屬性）：
      e.g.
        // 參數: cfg_section: config 的 section name.
        //       ctx_ptr: 屬於哪個 ctx.
        //       evq_ptr: 使用的 evq.
        //       src_cb: event 執行的 callback function.
        //       custom_data_ptr: callback function 的引數 (optional).
        //       custom_data_size: callback function 的引數大小 (byte) (optional).
        // config 格式: topic_name: 關注的 topic name.
        //              listen_ip: 此 src 的 ip.
        //              listen_port: 此 src 的 port.
        //              is_hot_failover_recv_mode:
        //                true: hot failover 接收資料模式,
        //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_SRC_DATA event.
        //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_SRC_DATA event.
        mws_src_t obj(std::string cfg_section,
                      mws_ctx_t* ctx_ptr,
                      mws_evq_t* evq_ptr,
                      callback_t src_cb,
                      void* custom_data_ptr = NULL,
                      const size_t custom_data_size = 0);
    ● mws_src_t 如何發送 message 給 rcv：
      e.g.
        // 回傳值 0: 表示成功 (只要傳送一個 receiver 成功則為 0).
        // 回傳值 -1: 表示失敗 (傳送全部 receiver 失敗則為 -1).
        // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
        // 參數 size_t len: 要傳送的 message 的長度 (byte).
        int i = obj.mws_src_send(const char* msg_ptr,
                                 size_t len);
    ● mws_src_t 如何以 hot failover 模式發送 message 給 rcv：
      e.g.
        // 回傳值 0: 表示成功 (只要傳送一個 receiver 成功則為 0).
        // 回傳值 -1: 表示失敗 (傳送全部 receiver 失敗則為 -1).
        // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
        // 參數 size_t len: 要傳送的 message 的長度 (byte).
        // 參數 seq_num: hot failover send 時所使用的序號.
        int i = obj.mws_hf_src_send(const char* msg_ptr,
                                    size_t len,
                                    uint64_t seq_num);
    ● mws_src_t 如何取得自己的 cfg section name：
      e.g.
        // 回傳值: src 的 cfg section name.
        // 參數: 無.
        std::string s = obj.mws_get_cfg_section();
    ------------------------------------------------------------

  － 關於 rcv 及 rcv attribute：
    ------------------------------------------------------------
    ● rcv attribute type：mws_rcv_attr_t
    ● rcv attribute 內容：
      1. topic_name.
      2. sess_addr_pair_XX.
      3. is_hot_failover_recv_mode.
    ● mws_rcv_attr_t 變數宣告方式：
      e.g.
        // 參數: cfg_section: config 的 section name.
        // config 格式: topic_name: 關注的 topic name.
        //              sess_addr_pair_XX: 此 rcv 所負責的所有 session 的 socket (listen & connect) 組合, XX 從01到99.
        //              is_hot_failover_recv_mode:
        //                true: hot failover 接收資料模式,
        //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_MSG_DATA event.
        //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_MSG_DATA event.
        mws_rcv_attr_t obj(std::string cfg_section);
    ● mws_rcv_attr_t 變數內容修改方式：
      e.g.
        // 回傳值: 沒有.
        // 參數: attr_name: attribute member 的名稱.
        // 參數: attr_value: attribute member 的值.
        obj.mws_modify_rcv_attr(std::string attr_name,
                                std::string attr_value);
    ------------------------------------------------------------
    ● rcv 注意事項：宣告一個 rcv 時需要綁定一個 ctx 和 evq。
    ● rcv type：mws_rcv_t
    ● mws_rcv_t 變數宣告方式（可在程式中宣告 mws_rcv_t 變數之前調整屬性）：
      e.g.
        // 參數: mws_rcv_attr: receiver attribute 物件.
        //       ctx_ptr: 屬於哪個 ctx.
        //       evq_ptr: 使用的 evq.
        //       rcv_cb: event 執行的 callback function.
        //       custom_data_ptr: callback function 的引數 (optional).
        //       custom_data_size: callback function 的引數大小 (byte) (optional).
        mws_rcv_t obj(mws_rcv_attr_t mws_rcv_attr,
                      mws_ctx_t* ctx_ptr,
                      mws_evq_t* evq_ptr,
                      callback_t rcv_cb,
                      void* custom_data_ptr = NULL,
                      const size_t custom_data_size = 0);
    ● mws_rcv_t 變數宣告方式（不可在程式中調整屬性）：
      e.g.
        // 參數: cfg_section: config 的 section name.
        //       ctx_ptr: 屬於哪個 ctx.
        //       evq_ptr: 使用的 evq.
        //       rcv_cb: event 執行的 callback function.
        //       custom_data_ptr: callback function 的引數 (optional).
        //       custom_data_size: callback function 的引數大小 (byte) (optional).
        // config 格式: topic_name: 關注的 topic name.
        //              sess_addr_pair_XX: 此 rcv 所負責的所有 session 的 socket (listen & connect) 組合, XX 從01到99.
        //              is_hot_failover_recv_mode:
        //                true: hot failover 接收資料模式,
        //                      message 的 sequence number 小於等於 max_seq_num 將會被忽略而不會觸發 MWS_MSG_DATA event.
        //                false: 非 hot failover 接收資料模式, 任何 message 都會觸發 MWS_MSG_DATA event.
        mws_rcv_t obj(std::string cfg_section,
                      mws_ctx_t* ctx_ptr,
                      mws_evq_t* evq_ptr,
                      callback_t rcv_cb,
                      void* custom_data_ptr = NULL,
                      const size_t custom_data_size = 0);
    ● mws_rcv_t 如何發送 message 給 src：
      e.g.
        // 回傳值 0: 表示成功 (只要傳送一個 source 成功則為 0).
        // 回傳值 -1: 表示失敗 (傳送全部 source 失敗則為 -1).
        // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
        // 參數 size_t len: 要傳送的 message 的長度 (byte).
        int i = obj.mws_rcv_send(const char* msg_ptr,
                                 size_t len);
    ● mws_rcv_t 如何以 hot failover 模式發送 message 給 src：
      e.g.
        // 回傳值 0: 表示成功 (只要傳送一個 source 成功則為 0).
        // 回傳值 -1: 表示失敗 (傳送全部 source 失敗則為 -1).
        // 參數 msg_ptr: 指向要傳送的 message 的 buffer 的指標.
        // 參數 size_t len: 要傳送的 message 的長度 (byte).
        // 參數 seq_num: hot failover send 時所使用的序號.
        int i = obj.mws_hf_rcv_send(const char* msg_ptr,
                                    size_t len,
                                    uint64_t seq_num);
    ● mws_rcv_t 如何取得自己的 cfg section name：
      e.g.
        // 回傳值: rcv 的 cfg section name.
        // 參數: 無.
        std::string s = obj.mws_get_cfg_section();
    ● mws_rcv_t 如何知道自己有多少個 session pair 的設定：
      e.g.
        // 回傳值: rcv 的 session pair 數量.
        // 參數: 無.
        size_t i = obj.mws_get_num_of_rcv_sessions();

    ------------------------------------------------------------
  － 關於 reactor only ctx 及 reactor only ctx attribute：

    ------------------------------------------------------------
  － 關於 endianness：


＊ Source Event 說明：
  －
  －

＊ Receiver Event 說明：
  －
  －

＊ 其他：
  － mws_log.h 為函式庫內部使用，AP programmer 不需要使用。