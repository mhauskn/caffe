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
}

template <typename Dtype>
void ExperienceDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // Initialize DB
  leveldb::DB* db_temp;
  leveldb::Options options = GetLevelDBOptions();
  options.create_if_missing = true;
  LOG(INFO) << "Opening leveldb " <<
      this->layer_param_.experience_param().source();
  leveldb::Status status = leveldb::DB::Open(
      options, this->layer_param_.experience_param().source(), &db_temp);
  CHECK(status.ok()) << "Failed to open leveldb "
                     << this->layer_param_.experience_param().source()
                     << std::endl
                     << status.ToString();
  db_.reset(db_temp);
  iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
  iter_->SeekToFirst();

  // Read the layer params, and use them to initialize the top blob.
  const int batch_size  = this->layer_param_.experience_param().batch_size();
  const int channels    = this->layer_param_.experience_param().channels();
  const int height      = this->layer_param_.experience_param().height();
  const int width       = this->layer_param_.experience_param().width();
  const int num_actions = this->layer_param_.experience_param().num_actions();

  (*top)[0]->Reshape(batch_size, channels, height, width);

  // Need to do this to please BasePrefetchingDataLayer::LayerSetup
  this->prefetch_data_.Reshape(1,1,1,1);
  this->prefetch_label_.Reshape(1,1,1,1);

  // Reshape the prefetch blobs
  this->prefetch_states_.Reshape(batch_size, channels, height, width);
  this->prefetch_new_states_.Reshape(batch_size, channels, height, width);

  // Reshape the action and reward vectors
  actions_.reset(new vector<int>(batch_size));
  rewards_.reset(new vector<float>(batch_size));

  LOG(INFO) << "output data size: " << (*top)[0]->num() << ","
      << (*top)[0]->channels() << "," << (*top)[0]->height() << ","
      << (*top)[0]->width();

  // label
  if (this->output_labels_) {
    (*top)[1]->Reshape(batch_size, num_actions, 1, 1);
    labels_.reset(new Blob<Dtype>(batch_size, num_actions, 1, 1));
  }
  // datum size
  this->datum_channels_ = channels;
  this->datum_height_ = height;
  this->datum_width_ = width;
  this->datum_size_ = channels * height * width;
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ExperienceDataLayer<Dtype>::InternalThreadEntry() {
  // Database may not be populated yet. Return if this is the case.
  if (!iter_->Valid()) {
    return;
  }
  Experience experience;
  const int batch_size = this->layer_param_.experience_param().batch_size();
  CHECK(this->prefetch_states_.count());
  CHECK(this->prefetch_new_states_.count());
  Dtype* state_data = prefetch_states_.mutable_cpu_data();
  Dtype* new_state_data = prefetch_new_states_.mutable_cpu_data();
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
    (*actions_)[item_id] = experience.action();
    (*rewards_)[item_id] = experience.reward();

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

  // Database may be newly populated
  if (!iter_->Valid()) {
    iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
    iter_->SeekToFirst();

    if (!iter_->Valid()) {
      LOG(FATAL) << "Cannot complete forward call while db is empty.";
    } else {
      this->CreatePrefetchThread();
      this->JoinPrefetchThread();
    }
  }

  if (first_forward_) {
    caffe_copy(prefetch_new_states_.count(), prefetch_new_states_.cpu_data(),
               (*top)[0]->mutable_cpu_data());
    first_forward_ = false;
  } else {
    caffe_copy(prefetch_states_.count(), prefetch_states_.cpu_data(),
               (*top)[0]->mutable_cpu_data());
    if (this->output_labels_) {
      caffe_copy(labels_->count(), labels_->cpu_data(),
                 (*top)[1]->mutable_cpu_data());
    }
    // Start a new prefetch thread
    this->CreatePrefetchThread();
  }
}

template <typename Dtype>
void ExperienceDataLayer<Dtype>::ComputeDataMean() {
  LOG(INFO) << "Computing Data Mean";
  leveldb::Iterator* it = NULL;
  leveldb::ReadOptions read_options;
  read_options.fill_cache = false;
  it = db_->NewIterator(read_options);
  it->SeekToFirst();

  if (!iter_->Valid()) {
    LOG(FATAL) << "Cannot compute data mean over an empty database!";
  }

  Experience experience;
  BlobProto sum_blob;
  int count = 0;
  // load first datum
  experience.ParseFromString(it->value().ToString());
  sum_blob.set_num(1);
  const Datum& datum = experience.state();
  sum_blob.set_channels(datum.channels());
  sum_blob.set_height(datum.height());
  sum_blob.set_width(datum.width());
  const int data_size = datum.channels() * datum.height() * datum.width();
  int size_in_datum = std::max<int>(datum.data().size(),
                                    datum.float_data_size());
  for (int i = 0; i < size_in_datum; ++i) {
    sum_blob.add_data(0.);
  }
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    // just a dummy operation
    experience.ParseFromString(it->value().ToString());
    const Datum& state = experience.state();
    const string& data = state.data();
    // We omit the next-state from averaging as it will be a previous
    // state of some other experience.
    size_in_datum = std::max<int>(state.data().size(),
                                  state.float_data_size());
    CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
        size_in_datum;
    if (data.size() != 0) {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
      }
    } else {
      for (int i = 0; i < size_in_datum; ++i) {
        sum_blob.set_data(i, sum_blob.data(i) +
                          static_cast<float>(state.float_data(i)));
      }
    }
    ++count;
    if (count % 10000 == 0) {
      LOG(ERROR) << "Processed " << count << " files.";
    }
  }

  if (count % 10000 != 0) {
    LOG(ERROR) << "Processed " << count << " files.";
  }
  for (int i = 0; i < sum_blob.data_size(); ++i) {
    sum_blob.set_data(i, sum_blob.data(i) / count);
  }
  this->data_mean_.FromProto(sum_blob);
}

INSTANTIATE_CLASS(ExperienceDataLayer);

}  // namespace caffe
