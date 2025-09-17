#include <numa.h>
#include <iostream>

#include "Common.h"
#include "Config.h"
#include "DSM.h"
#include "cache/node_wr.h"
#include "sherman_wrapper.h"
#include "smart/smart_wrapper.h"
#include "tree/leanstore_tree.h"
#include "tree_api.h"

// Derived from test/newbench.cpp and ported to Pentathlon's benchmark
// framework.
//
// To run this, memcached server and hugepage sysctl should be set up the same
// as original newbench.

// from run.sh
#if defined(DEX_COMPILE_DEX)
int kNodeCount = 2;
#elif defined(DEX_COMPILE_SHERMAN)
int kNodeCount = 1;
#else
#error please define DEX_COMPILE_DEX or DEX_COMPILE_SHERMAN
#endif
int kInsertRatio = 0;
int totalThreadCount = 2; // threads=(0 2 18 36 72 108 144)
int memThreadCount = 4;
int cache_mb = 128; // cache=(0 64 128 256 512 1024)
int bulk_load_num = 50 * 1000 * 1000;
int warmup_num = 10 * 1000 * 1000;
int op_num = 50 * 1000 * 1000;
int kMaxThread = 1;

int kKeySpace =
    bulk_load_num + ceil((op_num + warmup_num) * (kInsertRatio / 100.0)) + 1000;
int threadKSpace = kKeySpace / totalThreadCount;
int CNodeCount = (totalThreadCount % kMaxThread == 0)
                     ? (totalThreadCount / kMaxThread)
                     : (totalThreadCount / kMaxThread + 1);
double rpc_rate = 0.99;
double admission_rate = 0.1;

int thread_op_num = op_num / totalThreadCount;
int thread_warmup_num = warmup_num / totalThreadCount;

bool partitioned = false;

inline Key to_key(uint64_t k) {
  return (CityHash64((char *)&k, sizeof(k)) + 1) % kKeySpace;
}

long get_long_from_env(const char *env_name, long default_value) {
    char *env = getenv(env_name);
    if (env) {
        return strtol(env, NULL, 10);
    } else {
        return default_value;
    }
}

extern "C" {
void *pth_bm_target_create() {
  // bindCore(0);
  // numa_set_preferred(0);

  cache_mb = get_long_from_env("DEX_CACHE_MB", 128);
  std::cout << "DEX/Sherman cache size: " << cache_mb << " MB" << std::endl;

  DSMConfig config;
  config.machineNR = kNodeCount;
  config.memThreadCount = memThreadCount;
  config.computeNR = CNodeCount;
#if defined(DEX_COMPILE_DEX)
  config.index_type = 0; // DEX
#elif defined(DEX_COMPILE_SHERMAN)
  config.index_type = 1; // Sherman
#endif

  auto dsm = DSM::getInstance(config);
  cachepush::global_dsm_ = dsm;

  dsm->registerThread();

  tree_api<Key, Value> *tree = NULL;

  // correspond to generate_index()
  switch (config.index_type) {
  case 0: { // DEX
    // First set partition info
    int cluster_num = CNodeCount;
    std::vector<Key> sharding;
    sharding.push_back(std::numeric_limits<Key>::min());
    for (int i = 0; i < cluster_num - 1; ++i) {
      // sharding.push_back((threadKSpace * kMaxThread) + sharding[i]);
      sharding.push_back(std::numeric_limits<Key>::min());
      std::cout << "CNode " << i << ", left bound = " << sharding[i]
                << ", right bound = " << sharding[i + 1] << std::endl;
    }
    sharding.push_back(std::numeric_limits<Key>::max());
    std::cout << "CNode " << cluster_num - 1
              << ", left bound = " << sharding[cluster_num - 1]
              << ", right bound = " << sharding[cluster_num] << std::endl;
    assert(sharding.size() == cluster_num + 1);
    tree = new cachepush::BTree<Key, Value>(
        dsm, 0, cache_mb, rpc_rate, admission_rate, sharding, cluster_num);
    partitioned = true;
    tree->set_bound(std::numeric_limits<Key>::min(), std::numeric_limits<Key>::max());
    std::vector<uint64_t> bound = {std::numeric_limits<Key>::max()};
    tree->set_shared(bound);
    tree->get_basic();
  } break;
  case 1: // Sherman
    tree = new sherman_wrapper<Key, Value>(dsm, 0, cache_mb);
    // First insert one million ops to it to make sure the multi-thread
    // bulkloading can succeeds; otherwise, sherman has concurrency
    // bulkloading bug
    if (dsm->getMyNodeID() == 0) {
      for (uint64_t i = 1; i < 1024000; ++i) {
        tree->insert(to_key(i), i * 2);
      }
    }
    break;
  case 2: // SMART
    tree = new smart_wrapper<Key, Value>(dsm, 0, cache_mb);
    break;
  }

  dsm->resetThread();
  dsm->registerThread();
  tree->reset_buffer_pool(true);
  tree->get_newest_root();

  return tree;
}

void pth_bm_target_init_thread(void *target) {
  cachepush::global_dsm_->registerThread();
}

void pth_bm_target_destroy(void *target) {
  auto *tree = reinterpret_cast<tree_api<Key, Value> *>(target);
  delete tree;
}

void pth_bm_target_read(void *target, int key) {
  auto *tree = reinterpret_cast<tree_api<Key, Value> *>(target);
  unsigned long value = 0;
  tree->lookup(key, value);
}

void pth_bm_target_insert(void *target, int key) {
  auto *tree = reinterpret_cast<tree_api<Key, Value> *>(target);
  tree->insert(key, 0xdeadbeef);
}

void pth_bm_target_update(void *target, int key) {
  auto *tree = reinterpret_cast<tree_api<Key, Value> *>(target);
  tree->insert(key, 0xcafe0000);
}

void pth_bm_target_delete(void *target, int key) {
  auto *tree = reinterpret_cast<tree_api<Key, Value> *>(target);
  tree->remove(key);
}
}
