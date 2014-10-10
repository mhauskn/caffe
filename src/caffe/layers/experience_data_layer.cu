#include <vector>

#include "caffe/data_layers.hpp"

namespace caffe {

template <typename Dtype>
void ExperienceDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // First, join the thread
  this->JoinPrefetchThread();

  if (first_forward_) {
    caffe_copy(prefetch_new_states_.count(), prefetch_new_states_.cpu_data(),
               (*top)[0]->mutable_gpu_data());
    first_forward_ = false;
  } else {
    caffe_copy(prefetch_states_.count(), prefetch_states_.cpu_data(),
               (*top)[0]->mutable_gpu_data());
    if (this->output_labels_) {
      caffe_copy(labels_->count(), labels_->cpu_data(),
                 (*top)[1]->mutable_gpu_data());
    }
    // Start a new prefetch thread
    this->CreatePrefetchThread();
  }
}


INSTANTIATE_CLASS(ExperienceDataLayer);

}  // namespace caffe
