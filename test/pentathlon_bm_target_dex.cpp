#include <numa.h>

#include "Common.h"
#include "Config.h"
#include "DSM.h"
#include "cache/node_wr.h"
#include "tree_api.h"

extern "C" {
void *pth_bm_target_create() {
  // bindCore(0);
  // numa_set_preferred(0);

  DSMConfig config;
  config.machineNR = 1;
  config.memThreadCount = 4;
  config.computeNR = 1;
  config.index_type = 0; // DEX

  auto dsm = DSM::getInstance(config);
  cachepush::global_dsm_ = dsm;

  // TODO: kThreadCount

  // TODO: return tree_api
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
