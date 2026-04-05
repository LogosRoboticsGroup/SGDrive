export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/path/to/NAVSIM/dataset/maps"
export NAVSIM_EXP_ROOT="/path/to/NAVSIM/exp"
export NAVSIM_DEVKIT_ROOT="/path/to/NAVSIM/navsim-main"
export OPENSCENE_DATA_ROOT="/path/to/NAVSIM/dataset"

export PYTHONPATH="$(pwd):$(pwd)/internvl_chat:${PYTHONPATH}"

TRAIN_TEST_SPLIT=navmini
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_${TRAIN_TEST_SPLIT}
if [ ! -d "$CACHE_PATH" ]; then
  mkdir -p "$CACHE_PATH"
fi

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=$TRAIN_TEST_SPLIT \
    cache.cache_path=$CACHE_PATH