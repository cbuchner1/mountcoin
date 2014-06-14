// Copyright (c) 2012-2013 The Cryptonote developers
// Distributed under the MIT/X11 software license, see the accompanying
// file COPYING or http://www.opensource.org/licenses/mit-license.php.


#include <sstream>
#include <numeric>
#include <boost/utility/value_init.hpp>
#include <boost/interprocess/detail/atomic.hpp>
#include <boost/limits.hpp>
#include <boost/foreach.hpp>
#include "misc_language.h"
#include "include_base_utils.h"
#include "cryptonote_basic_impl.h"
#include "cryptonote_format_utils.h"
#include "file_io_utils.h"
#include "common/command_line.h"
#include "string_coding.h"
#include "storages/portable_storage_template_helper.h"

using namespace epee;

#include "crypto/oaes_lib.h"
extern "C" {
  #include "crypto/keccak.h"
}
#include "gpuminer.h"

static uint32_t *h_keccakOutput[8];
static uint32_t *h_AESKey1[8];
static uint32_t *h_AESKey2[8];
static uint32_t *h_abInput[8];
static uint32_t *h_HashOutput[8];

#define MEMORY         (1 << 21) // 2MB scratchpad
#define MEMORY64       (1 << 18) /* in 64 bit words */
#define ITER           (1 << 20)
#define AES_BLOCK_SIZE  16
#define AES_KEY_SIZE    32
#define INIT_SIZE_BLK   8
#define INIT_SIZE_BYTE (INIT_SIZE_BLK * AES_BLOCK_SIZE)

#pragma pack(push, 1)
union hash_state {
  uint8_t b[200];
  uint64_t w[25];
};
#pragma pack(pop)

#pragma pack(push, 1)
union cn_slow_hash_state
{
    union hash_state hs;
    struct
    {
        uint8_t k[64];
        uint8_t init[INIT_SIZE_BYTE];
    };
};
#pragma pack(pop)

//extern "C" void hash_permutation(union hash_state *state);
//extern "C" void hash_process(union hash_state *state, const uint8_t *buf, size_t count);

extern "C" void hash_extra_blake(const void *data, size_t length, char *hash);
extern "C" void hash_extra_groestl(const void *data, size_t length, char *hash);
extern "C" void hash_extra_jh(const void *data, size_t length, char *hash);
extern "C" void hash_extra_skein(const void *data, size_t length, char *hash);

int cryptonight_num_smx(int thr_id);

void cryptonight_cpu_init(int thr_id, int threads);

void cryptonight_cpu_hash_test_flex(int thr_id, int threads, uint32_t *h_keccakOutputData,
    uint32_t *h_keccakOutputExpandedAESKey1, uint32_t *h_keccakOutputExpandedAESKey2,
    uint32_t *h_abInput, uint32_t *h_hashOutput, int order, const int threadsperblock);

void cryptonight_gpu_hash_prep(int threads, int thr_id, const void* input, size_t len, int startnonce, int threads_total)
{
    OAES_CTX* aes_ctx;
    union cn_slow_hash_state state;

    aes_ctx = oaes_alloc();

    uint32_t *nonceptr = (uint32_t *)(((char *)input) + 39);
    uint32_t tmp = *nonceptr; 
    for(int i=0;i<threads;i++)
    {
        // prepare keccak
        *nonceptr = startnonce+i*threads_total;
        keccak1600((const uint8_t *)input, 76, (uint8_t*)&state.hs);
        memcpy(&(h_keccakOutput[thr_id])[i*34], &state.hs.b[64], 136);
        // gen aes keys	
        oaes_key_import_data(aes_ctx, &state.hs.b[32], AES_KEY_SIZE);
        memcpy(&(h_AESKey2[thr_id])[i*40], (uint32_t*)(((oaes_ctx *)aes_ctx)->key->exp_data), 160);
        oaes_key_import_data(aes_ctx, &state.hs.b[0], AES_KEY_SIZE);
        memcpy(&(h_AESKey1[thr_id])[i*40], (uint32_t*)(((oaes_ctx *)aes_ctx)->key->exp_data), 160);	
        // generate A+B
        memcpy(&(h_abInput[thr_id])[i*16], (uint32_t*)state.k, 64);
    }
    *nonceptr = tmp;

    oaes_free(&aes_ctx);
}

void cryptonight_gpu_hash_post(int threads, int thr_id, int i, void* output)
{
    union cn_slow_hash_state state;

    memcpy(state.hs.b, &(h_abInput[thr_id])[i*16], 64);
    memcpy(&state.hs.b[64], &(h_keccakOutput[thr_id])[i*34], 136);
    memcpy(state.init, &(h_HashOutput[thr_id])[i*32], INIT_SIZE_BYTE);
    keccakf((uint64_t*)&state, 24); /* hash_permutation */

    static void (*const extra_hashes[4])(const void *, size_t, char *) = {
        hash_extra_blake, hash_extra_groestl, hash_extra_jh, hash_extra_skein
    };

    extra_hashes[state.hs.b[0] & 3](&state, 200, (char*)output);
}
    
namespace cryptonote
{
  static void gpu_hash(int thr_id, const void *data, size_t length, char *hash, uint32_t nonce, int throughput, int threadsperblock, int threads_total)
  {
    cryptonight_gpu_hash_prep(throughput, thr_id, data, length, nonce, threads_total);

    cryptonight_cpu_hash_test_flex(thr_id, throughput, h_keccakOutput[thr_id],
                                   h_AESKey1[thr_id], h_AESKey2[thr_id],
                                   h_abInput[thr_id], h_HashOutput[thr_id], 0, threadsperblock);

  }

  static bool my_scanhash(int thr_id, block& b, uint64_t height, uint32_t &nonce, int threads_total, difficulty_type local_diff, std::atomic<uint64_t> &hashes, int threadsperblock)
  {
    static bool init[8] = {false, false, false, false, false, false, false, false};
    static int smx = 0;
    
    if (!init[thr_id])
    {
      smx = cryptonight_num_smx(thr_id);
      int throughput = threadsperblock*smx;
      
      LOG_PRINT_L0("GPU Miner thread ["<< thr_id << "] using " << throughput << " threads (" << smx << " x " << threadsperblock << ")" );

      cryptonight_cpu_init(thr_id, throughput);
      
      h_keccakOutput[thr_id] = (uint32_t*)malloc(136*throughput);
      h_AESKey1[thr_id] = (uint32_t*)malloc(160*throughput);
      h_AESKey2[thr_id] = (uint32_t*)malloc(160*throughput);
      h_abInput[thr_id] = (uint32_t*)malloc(64*throughput);
      h_HashOutput[thr_id] = (uint32_t*)malloc(128*throughput);

      init[thr_id] = true;
    }
  
    int throughput = threadsperblock*smx;
    bool found = false;

    b.nonce = nonce;
    blobdata bd = get_block_hashing_blob(b);

    crypto::hash h;
    gpu_hash(thr_id, bd.data(), bd.size(), reinterpret_cast<char *>(&h), nonce, throughput, threadsperblock, threads_total);
    
    for (int i=0; i < throughput; i++)
    {
      crypto::hash gpuhash;
      cryptonight_gpu_hash_post(throughput, thr_id, i, reinterpret_cast<char *>(&gpuhash));

#if 0
      // validation with the CPU
      b.nonce = nonce + i*threads_total;
      bd = get_block_hashing_blob(b);
      crypto::cn_slow_hash(bd.data(), bd.size(), h);
     
      if (memcmp(reinterpret_cast<char *>(&h), reinterpret_cast<char *>(&gpuhash), 32))
      {
        fprintf(stderr, "CPU and GPU hash for nonce $%08x do not agree!\n", nonce+i*threads_total);
      }
      else
      {
        fprintf(stderr, "CPU and GPU hash for nonce $%08x agree!\n", nonce+i*threads_total);
      }
#endif
      found = check_hash(gpuhash, local_diff);
      if (found)
      {
        b.nonce = nonce + i*threads_total;
        break;
      }
    }
    
    nonce+=threads_total*throughput;
    hashes+=throughput;
    
    return found;
  }

  namespace
  {
    const command_line::arg_descriptor<std::string> arg_extra_messages =  {"extra-messages-file", "Specify file for extra messages to include into coinbase transactions", "", true};
    const command_line::arg_descriptor<std::string> arg_start_gpumining =    {"start-gpumining", "Specify wallet address to mining for", "", true};
    const command_line::arg_descriptor<uint32_t>      arg_gpumining_threads =  {"gpumining-threads", "Specify mining threads count", 0, true};
    const command_line::arg_descriptor<uint32_t>      arg_gpumining_threadsperblock =  {"gpumining-threadsperblock", "Specify number of threads per CUDA threadblock", 64, true};
  }


  gpuminer::gpuminer(i_miner_handler* phandler):m_stop(1),
    m_template(boost::value_initialized<block>()),
    m_template_no(0),
    m_diffic(0),
    m_thread_index(0),
    m_phandler(phandler),
    m_height(0),
    m_pausers_count(0), 
    m_threads_total(0),
    m_threadsperblock(64),
    m_starter_nonce(0), 
    m_last_hr_merge_time(0),
    m_hashes(0),
    m_do_print_hashrate(false),
    m_do_mining(false),
    m_current_hash_rate(0)
  {

  }
  //-----------------------------------------------------------------------------------------------------
  gpuminer::~gpuminer()
  {
    stop();
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::set_block_template(const block& bl, const difficulty_type& di, uint64_t height)
  {
    CRITICAL_REGION_LOCAL(m_template_lock);
    m_template = bl;
    m_diffic = di;
    m_height = height;
    ++m_template_no;
    m_starter_nonce = crypto::rand<uint32_t>();
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::on_block_chain_update()
  {
    if(!is_mining())
      return true;

    return request_block_template();
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::request_block_template()
  {
    block bl = AUTO_VAL_INIT(bl);
    difficulty_type di = AUTO_VAL_INIT(di);
    uint64_t height = AUTO_VAL_INIT(height);
    cryptonote::blobdata extra_nonce; 
    if(m_extra_messages.size() && m_config.current_extra_message_index < m_extra_messages.size())
    {
      extra_nonce = m_extra_messages[m_config.current_extra_message_index];
    }

    if(!m_phandler->get_block_template(bl, m_mine_address, di, height, extra_nonce))
    {
      LOG_ERROR("Failed to get_block_template(), stopping mining");
      return false;
    }
    set_block_template(bl, di, height);
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::on_idle()
  {
    m_update_block_template_interval.do_call([&](){
      if(is_mining())request_block_template();
      return true;
    });

    m_update_merge_hr_interval.do_call([&](){
      merge_hr();
      return true;
    });
    
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::do_print_hashrate(bool do_hr)
  {
    m_do_print_hashrate = do_hr;
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::merge_hr()
  {
    if(m_last_hr_merge_time && is_mining())
    {
      m_current_hash_rate = m_hashes * 1000 / ((misc_utils::get_tick_count() - m_last_hr_merge_time + 1));
      CRITICAL_REGION_LOCAL(m_last_hash_rates_lock);
      m_last_hash_rates.push_back(m_current_hash_rate);
      if(m_last_hash_rates.size() > 19)
        m_last_hash_rates.pop_front();
      if(m_do_print_hashrate)
      {
        uint64_t total_hr = std::accumulate(m_last_hash_rates.begin(), m_last_hash_rates.end(), 0);
        float hr = static_cast<float>(total_hr)/static_cast<float>(m_last_hash_rates.size());
        std::cout << "GPU hashrate: " << std::setprecision(4) << std::fixed << hr << ENDL;
      }
    }
    m_last_hr_merge_time = misc_utils::get_tick_count();
    m_hashes = 0;
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::init_options(boost::program_options::options_description& desc)
  {
    command_line::add_arg(desc, arg_extra_messages);
    command_line::add_arg(desc, arg_start_gpumining);
    command_line::add_arg(desc, arg_gpumining_threads);
    command_line::add_arg(desc, arg_gpumining_threadsperblock);
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::init(const boost::program_options::variables_map& vm)
  {
    if(command_line::has_arg(vm, arg_extra_messages))
    {
      std::string buff;
      bool r = file_io_utils::load_file_to_string(command_line::get_arg(vm, arg_extra_messages), buff);
      CHECK_AND_ASSERT_MES(r, false, "Failed to load file with extra messages: " << command_line::get_arg(vm, arg_extra_messages));
      std::vector<std::string> extra_vec;
      boost::split(extra_vec, buff, boost::is_any_of("\n"), boost::token_compress_on );
      m_extra_messages.resize(extra_vec.size());
      for(size_t i = 0; i != extra_vec.size(); i++)
      {
        string_tools::trim(extra_vec[i]);
        if(!extra_vec[i].size())
          continue;
        std::string buff = string_encoding::base64_decode(extra_vec[i]);
        if(buff != "0")
          m_extra_messages[i] = buff;
      }
      m_config_folder_path = boost::filesystem::path(command_line::get_arg(vm, arg_extra_messages)).parent_path().string();
      m_config = AUTO_VAL_INIT(m_config);
      epee::serialization::load_t_from_json_file(m_config, m_config_folder_path + "/" + MINER_CONFIG_FILE_NAME);
      LOG_PRINT_L0("Loaded " << m_extra_messages.size() << " extra messages, current index " << m_config.current_extra_message_index);
    }

    if(command_line::has_arg(vm, arg_start_gpumining))
    {
      if(!cryptonote::get_account_address_from_str(m_mine_address, command_line::get_arg(vm, arg_start_gpumining)))
      {
        LOG_ERROR("Target account address " << command_line::get_arg(vm, arg_start_gpumining) << " has wrong format, starting daemon canceled");
        return false;
      }
      m_threads_total = 1;
      m_do_mining = true;
      if(command_line::has_arg(vm, arg_gpumining_threads))
      {
        m_threads_total = command_line::get_arg(vm, arg_gpumining_threads);
      }
      if(command_line::has_arg(vm, arg_gpumining_threadsperblock))
      {
        m_threadsperblock = command_line::get_arg(vm, arg_gpumining_threadsperblock);
      }
    }

    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::is_mining()
  {
    return !m_stop;
  }
  //----------------------------------------------------------------------------------------------------- 
  bool gpuminer::start(const account_public_address& adr, size_t threads_count, const boost::thread::attributes& attrs)
  {
    m_mine_address = adr;
    m_threads_total = static_cast<uint32_t>(threads_count);
    m_starter_nonce = crypto::rand<uint32_t>();
    CRITICAL_REGION_LOCAL(m_threads_lock);
    if(is_mining())
    {
      LOG_ERROR("Starting gpuminer but it's already started");
      return false;
    }

    if(!m_threads.empty())
    {
      LOG_ERROR("Unable to start gpuminer because there are active mining threads");
      return false;
    }

    if(!m_template_no)
      request_block_template();//lets update block template

    boost::interprocess::ipcdetail::atomic_write32(&m_stop, 0);
    boost::interprocess::ipcdetail::atomic_write32(&m_thread_index, 0);

    for(size_t i = 0; i != threads_count; i++)
    {
      m_threads.push_back(boost::thread(attrs, boost::bind(&gpuminer::worker_thread, this)));
    }

    LOG_PRINT_L0("GPU Mining has started with " << threads_count << " threads, good luck!" )
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  uint64_t gpuminer::get_speed()
  {
    if(is_mining())
      return m_current_hash_rate;
    else
      return 0;
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::send_stop_signal()
  {
    boost::interprocess::ipcdetail::atomic_write32(&m_stop, 1);
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::stop()
  {
    send_stop_signal();
    CRITICAL_REGION_LOCAL(m_threads_lock);

    BOOST_FOREACH(boost::thread& th, m_threads)
      th.join();

    m_threads.clear();
    LOG_PRINT_L0("GPU Mining has been stopped, " << m_threads.size() << " finished" );
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::find_nonce_for_given_block(block& bl, const difficulty_type& diffic, uint64_t height)
  {
    for(; bl.nonce != std::numeric_limits<uint32_t>::max(); bl.nonce++)
    {
      crypto::hash h;
      get_block_longhash(bl, h, height);

      if(check_hash(h, diffic))
      {
        return true;
      }
    }
    return false;
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::on_synchronized()
  {
    if(m_do_mining)
    {
      boost::thread::attributes attrs;
      attrs.set_stack_size(THREAD_STACK_SIZE);

      start(m_mine_address, m_threads_total, attrs);
    }
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::pause()
  {
    CRITICAL_REGION_LOCAL(m_gpuminers_count_lock);
    ++m_pausers_count;
    if(m_pausers_count == 1 && is_mining())
      LOG_PRINT_L2("GPU MINING PAUSED");
  }
  //-----------------------------------------------------------------------------------------------------
  void gpuminer::resume()
  {
    CRITICAL_REGION_LOCAL(m_gpuminers_count_lock);
    --m_pausers_count;
    if(m_pausers_count < 0)
    {
      m_pausers_count = 0;
      LOG_PRINT_RED_L0("Unexpected gpuminer::resume() called");
    }
    if(!m_pausers_count && is_mining())
      LOG_PRINT_L2("GPU MINING RESUMED");
  }
  //-----------------------------------------------------------------------------------------------------
  bool gpuminer::worker_thread()
  {
    uint32_t th_local_index = boost::interprocess::ipcdetail::atomic_inc32(&m_thread_index);
    LOG_PRINT_L0("GPU Miner thread was started ["<< th_local_index << "]");
    log_space::log_singletone::set_thread_log_prefix(std::string("[gpuminer ") + std::to_string(th_local_index) + "]");
    uint32_t nonce = m_starter_nonce + th_local_index;
    uint64_t height = 0;
    difficulty_type local_diff = 0;
    uint32_t local_template_ver = 0;
    block b;
    while(!m_stop)
    {
      if(m_pausers_count)//anti split workaround
      {
        misc_utils::sleep_no_w(100);
        continue;
      }

      if(local_template_ver != m_template_no)
      {
        
        CRITICAL_REGION_BEGIN(m_template_lock);
        b = m_template;
        local_diff = m_diffic;
        height = m_height;
        CRITICAL_REGION_END();
        local_template_ver = m_template_no;
        nonce = m_starter_nonce + th_local_index;
      }

      if(!local_template_ver)//no any set_block_template call
      {
        LOG_PRINT_L2("Block template not set yet");
        epee::misc_utils::sleep_no_w(1000);
        continue;
      }

      if (my_scanhash(th_local_index, b, height, nonce, m_threads_total, local_diff, m_hashes, m_threadsperblock))
      {
        //we lucky!
        ++m_config.current_extra_message_index;
        LOG_PRINT_GREEN("Found block for difficulty: " << local_diff, LOG_LEVEL_0);
        if(!m_phandler->handle_block_found(b))
        {
          --m_config.current_extra_message_index;
        }else
        {
          //success update, lets update config
          epee::serialization::store_t_to_json_file(m_config, m_config_folder_path + "/" + MINER_CONFIG_FILE_NAME);
        }
      }
    }
    LOG_PRINT_L0("GPU Miner thread stopped ["<< th_local_index << "]");
    return true;
  }
  //-----------------------------------------------------------------------------------------------------
}

