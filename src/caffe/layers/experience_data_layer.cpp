#include <leveldb/db.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ExperienceDataLayer<Dtype>::~ExperienceDataLayer<Dtype>() {
  this->JoinPrefetchThread();
  // Destroy the database once done
  DestroyDB(this->layer_param_.data_param().source(), GetLevelDBOptions());
}

template <typename Dtype>
void ExperienceDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // Initialize DB
  leveldb::DB* db_temp;
  leveldb::Options options = GetLevelDBOptions();
  options.create_if_missing = false;
  LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.data_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << this->layer_param_.data_param().source() << std::endl
                     << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    while (skip-- > 0) {
      iter_->Next();
      if (!iter_->Valid()) {
        iter_->SeekToFirst();
      }
    }
  }
  // Read a data point, and use it to initialize the top blob.
  Experience experience;
  experience.ParseFromString(iter_->value().ToString());
  const Datum& datum = experience.state();

  (*top)[0]->Reshape(
      this->layer_param_.data_param().batch_size(), datum.channels(),
      datum.height(), datum.width());

  // Need to do this to please BasePrefetchingDataLayer::LayerSetup
  this->prefetch_data_.Reshape(1,1,1,1);
  this->prefetch_label_.Reshape(1,1,1,1);

  // Reshape the prefetch blobs
  this->prefetch_states_.Reshape(
      this->layer_param_.data_param().batch_size(),
      datum.channels(), datum.height(), datum.width());
  this->prefetch_new_states_.Reshape(
      this->layer_param_.data_param().batch_size(),
      datum.channels(), datum.height(), datum.width());

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // TODO(mhauskn): Reshape labels correctly
  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(this->layer_param_.data_param().batch_size(), 1, 1, 1);
    this->prefetch_label_.Reshape(this->layer_param_.data_param().batch_size(),
        1, 1, 1);
  }
  // datum size
  this->datum_channels_ = datum.channels();
  this->datum_height_ = datum.height();
  this->datum_width_ = datum.width();
  this->datum_size_ = datum.channels() * datum.height() * datum.width();
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ExperienceDataLayer<Dtype>::InternalThreadEntry() {
  Experience experience;
  const int batch_size = this->layer_param_.data_param().batch_size();
  Dtype* state_data = prefetch_states_.mutable_cpu_data();
  Dtype* new_state_data = prefetch_states_.mutable_cpu_data();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    CHECK(iter_);
    CHECK(iter_->Valid());
    experience.ParseFromString(iter_->value().ToString());

    // Apply data transformations (mirror, scale, crop...)
    this->data_transformer_.Transform(item_id, experience.state(),
                                      this->mean_, state_data);
    this->data_transformer_.Transform(item_id, experience.new_state(),
                                      this->mean_, new_state_data);

    // Store the actions and rewards
    actions.push_back(experience.action());
    rewards.push_back(experience.reward());

    // go to the next iter
    iter_->Next();
    if (!iter_->Valid()) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      iter_->SeekToFirst();
    }
  }
  first_forward_ = true;
}

template <typename Dtype>
void ExperienceDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();

  if (first_forward_) {
    caffe_copy(prefetch_new_states_.count(), prefetch_new_states_.cpu_data(),
               (*top)[0]->mutable_cpu_data());
    first_forward_ = false;
  } else {
    caffe_copy(prefetch_states_.count(), prefetch_states_.cpu_data(),
               (*top)[0]->mutable_cpu_data());
    // if (this->output_labels_) {
    //   caffe_copy(prefetch_label_.count(), prefetch_label_.cpu_data(),
    //              (*top)[1]->mutable_cpu_data());
    // }
    // Start a new prefetch thread
    this->CreatePrefetchThread();
  }
}

INSTANTIATE_CLASS(ExperienceDataLayer);

}  // namespace caffe
