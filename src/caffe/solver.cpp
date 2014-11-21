#include <cstdio>

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/data_layers.hpp"
#include "caffe/loss_layers.hpp"

#include "leveldb/write_batch.h"
#include <opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param)
    : net_() {
  Init(param);
}

template <typename Dtype>
Solver<Dtype>::Solver(const string& param_file)
    : net_() {
  SolverParameter param;
  ReadProtoFromTextFile(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  LOG(INFO) << "Initializing solver from parameters: " << std::endl
            << param.DebugString();
  param_ = param;
  if (param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();
  InitTestNets();
  LOG(INFO) << "Solver scaffolding done.";
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG(INFO) << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG(INFO) << "Creating training net from train_net file: "
              << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG(INFO) << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG(INFO) << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
  }
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  net_.reset(new Net<Dtype>(net_param));
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    test_nets_[i].reset(new Net<Dtype>(net_params[i]));
  }
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << net_->name();
  PreSolve();

  iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
  // Remember the initial iter_ value; will be non-zero if we loaded from a
  // resume_file above.
  const int start_iter = iter_;

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  for (; iter_ < param_.max_iter(); ++iter_) {
    // Save a snapshot if needed.
    if (param_.snapshot() && iter_ > start_iter &&
        iter_ % param_.snapshot() == 0) {
      Snapshot();
    }

    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())) {
      TestAll();
    }

    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    Dtype loss = net_->ForwardBackward(bottom_vec);
    if (display) {
      LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG(INFO) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }

    ComputeUpdateValue();
    net_->Update();
  }
  // Always save a snapshot after optimization, unless overridden by setting
  // snapshot_after_train := false.
  if (param_.snapshot_after_train()) { Snapshot(); }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    Dtype loss;
    net_->Forward(bottom_vec, &loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << loss;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  LOG(INFO) << "Optimization Done.";
}


template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0; test_net_id < test_nets_.size(); ++test_net_id) {
    Test(test_net_id);
  }
}


template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  // We need to set phase to test before running.
  Caffe::set_phase(Caffe::TEST);
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(bottom_vec, &iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const string& output_name = test_net->blob_names()[
        test_net->output_blob_indices()[test_score_output_id[i]]];
    const Dtype loss_weight =
        test_net->blob_loss_weights()[test_net->output_blob_indices()[i]];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
        << mean_score << loss_msg_stream.str();
  }
  Caffe::set_phase(Caffe::TRAIN);
}


template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  NetParameter net_param;
  // For intermediate results, we will also dump the gradient values.
  net_->ToProto(&net_param, param_.snapshot_diff());
  string filename(param_.snapshot_prefix());
  string model_filename, snapshot_filename;
  const int kBufferSize = 20;
  char iter_str_buffer[kBufferSize];
  snprintf(iter_str_buffer, kBufferSize, "_iter_%d", iter_);
  filename += iter_str_buffer;
  model_filename = filename + ".caffemodel";
  LOG(INFO) << "Snapshotting to " << model_filename;
  WriteProtoToBinaryFile(net_param, model_filename.c_str());
  SolverState state;
  SnapshotSolverState(&state);
  state.set_iter(iter_);
  state.set_learned_net(model_filename);
  snapshot_filename = filename + ".solverstate";
  LOG(INFO) << "Snapshotting solver state to " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  SolverState state;
  NetParameter net_param;
  ReadProtoFromBinaryFile(state_file, &state);
  if (state.has_learned_net()) {
    ReadProtoFromBinaryFile(state.learned_net().c_str(), &net_param);
    net_->CopyTrainedLayersFrom(net_param);
  }
  iter_ = state.iter();
  RestoreSolverState(state);
}


// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
// where base_lr, gamma, step and power are defined in the solver parameter
// protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    int current_step = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), current_step);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}


template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  history_.clear();
  update_.clear();
  temp_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const Blob<Dtype>* net_param = net_params[i].get();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(
        net_param->num(), net_param->channels(), net_param->height(),
        net_param->width())));
  }
}


template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                history_[param_id]->mutable_cpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // Compute the value to history, and then copy them to the blob's diff.
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                history_[param_id]->mutable_gpu_data());
      // copy
      caffe_copy(net_params[param_id]->count(),
          history_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(SolverState* state) {
  state->clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state->add_history();
    history_[i]->ToProto(history_blob);
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverState(const SolverState& state) {
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}

template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype momentum = this->param_.momentum();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->cpu_data(),
          this->update_[param_id]->mutable_cpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              this->temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // update history
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->cpu_diff(), momentum,
                this->history_[param_id]->mutable_cpu_data());

      // compute udpate: step back then over step
      caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->cpu_data(), -momentum,
          this->update_[param_id]->mutable_cpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      // save history momentum for stepping back
      caffe_copy(net_params[param_id]->count(),
          this->history_[param_id]->gpu_data(),
          this->update_[param_id]->mutable_gpu_data());

      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              this->temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // update history
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
                net_params[param_id]->gpu_diff(), momentum,
                this->history_[param_id]->mutable_gpu_data());

      // compute udpate: step back then over step
      caffe_gpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
          this->history_[param_id]->gpu_data(), -momentum,
          this->update_[param_id]->mutable_gpu_data());

      // copy
      caffe_copy(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AdaGradSolver<Dtype>::ComputeUpdateValue() {
  vector<shared_ptr<Blob<Dtype> > >& net_params = this->net_->params();
  vector<float>& net_params_lr = this->net_->params_lr();
  vector<float>& net_params_weight_decay = this->net_->params_weight_decay();
  // get the learning rate
  Dtype rate = this->GetLearningRate();
  Dtype delta = this->param_.delta();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  switch (Caffe::mode()) {
  case Caffe::CPU:
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else if (regularization_type == "L1") {
          caffe_cpu_sign(net_params[param_id]->count(),
              net_params[param_id]->cpu_data(),
              this->temp_[param_id]->mutable_cpu_data());
          caffe_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->cpu_data(),
              net_params[param_id]->mutable_cpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // compute square of gradient in update
      caffe_powx(net_params[param_id]->count(),
          net_params[param_id]->cpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_cpu_data());

      // update history
      caffe_add(net_params[param_id]->count(),
          this->update_[param_id]->cpu_data(),
          this->history_[param_id]->cpu_data(),
          this->history_[param_id]->mutable_cpu_data());

      // prepare update
      caffe_powx(net_params[param_id]->count(),
                this->history_[param_id]->cpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_cpu_data());

      caffe_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_cpu_data());

      caffe_div(net_params[param_id]->count(),
                net_params[param_id]->cpu_diff(),
                this->update_[param_id]->cpu_data(),
                this->update_[param_id]->mutable_cpu_data());

      // scale and copy
      caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->cpu_data(), Dtype(0),
          net_params[param_id]->mutable_cpu_diff());
    }
    break;
  case Caffe::GPU:
#ifndef CPU_ONLY
    for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      Dtype local_rate = rate * net_params_lr[param_id];
      Dtype local_decay = weight_decay * net_params_weight_decay[param_id];

      if (local_decay) {
        if (regularization_type == "L2") {
          // add weight decay
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              net_params[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else if (regularization_type == "L1") {
          caffe_gpu_sign(net_params[param_id]->count(),
              net_params[param_id]->gpu_data(),
              this->temp_[param_id]->mutable_gpu_data());
          caffe_gpu_axpy(net_params[param_id]->count(),
              local_decay,
              this->temp_[param_id]->gpu_data(),
              net_params[param_id]->mutable_gpu_diff());
        } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
        }
      }

      // compute square of gradient in update
      caffe_gpu_powx(net_params[param_id]->count(),
          net_params[param_id]->gpu_diff(), Dtype(2),
          this->update_[param_id]->mutable_gpu_data());

      // update history
      caffe_gpu_add(net_params[param_id]->count(),
          this->update_[param_id]->gpu_data(),
          this->history_[param_id]->gpu_data(),
          this->history_[param_id]->mutable_gpu_data());

      // prepare update
      caffe_gpu_powx(net_params[param_id]->count(),
                this->history_[param_id]->gpu_data(), Dtype(0.5),
                this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_add_scalar(net_params[param_id]->count(),
                delta, this->update_[param_id]->mutable_gpu_data());

      caffe_gpu_div(net_params[param_id]->count(),
                net_params[param_id]->gpu_diff(),
                this->update_[param_id]->gpu_data(),
                this->update_[param_id]->mutable_gpu_data());

      // scale and copy
      caffe_gpu_axpby(net_params[param_id]->count(), local_rate,
          this->update_[param_id]->gpu_data(), Dtype(0),
          net_params[param_id]->mutable_gpu_diff());
    }
#else
    NO_GPU;
#endif
    break;
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void AtariSolver<Dtype>::PreSolve() {
  SGDSolver<Dtype>::PreSolve();
  // Load the ROM file
  ale_.loadROM(this->param_.atari_param().rom());
  gamma_ = Dtype(this->param_.atari_param().gamma());
  epsilon_ = Dtype(1);
  rescale_reward_ = this->param_.atari_param().rescale_reward();
  memory_size_ = this->param_.atari_param().replay_memory_size();
  LOG(INFO) << "Minimum Action Set Size: " << ale_.getMinimalActionSet().size();
  LOG(INFO) << "Legal Action Set Size: " << ale_.getLegalActionSet().size();

  shared_ptr<BaseDataLayer<Dtype> > data_layer =
      boost::dynamic_pointer_cast<BaseDataLayer<Dtype> >
      (this->net_->layers()[0]);
  CHECK(data_layer) <<
      "Input Layer to the Atari Train Net must be a DataLayer.";
  target_channels_ = data_layer->datum_channels();
  target_width_ = data_layer->datum_width();
  target_height_ = data_layer->datum_height();
}

template <typename Dtype>
void AtariSolver<Dtype>::Solve(const char* resume_file) {
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Solving " << this->net_->name();
  this->PreSolve();

  this->iter_ = 0;
  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    this->Restore(resume_file);
  }
  // Remember the initial iter_ value; will be non-zero if we loaded from a
  // resume_file above.
  const int start_iter = this->iter_;

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  vector<Blob<Dtype>*> bottom_vec;
  Dtype avg_loss(0);
  int loss_count = 0;
  for (; this->iter_ < this->param_.max_iter(); ++this->iter_) {
    // Save a snapshot if needed.
    if (this->param_.snapshot() && this->iter_ > start_iter &&
        this->iter_ % this->param_.snapshot() == 0) {
      this->Snapshot();
    }

    if (this->param_.test_interval() &&
        this->iter_ % this->param_.test_interval() == 0) {
      this->PlayAtari(0);
    }

    const bool display = this->param_.display() &&
        this->iter_ % this->param_.display() == 0;
    this->net_->set_debug_info(display && this->param_.debug_info());

    Dtype loss = ForwardBackward(bottom_vec);
    avg_loss += loss;
    loss_count++;
    if (display) {
      LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss
                << ", avg_loss = " << avg_loss / loss_count;
      avg_loss = Dtype(0);
      loss_count = 0;
    }

    this->ComputeUpdateValue();
    this->net_->Update();
  }
  // Always save a snapshot after optimization, unless overridden by setting
  // snapshot_after_train := false.
  if (this->param_.snapshot_after_train()) { this->Snapshot(); }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    Dtype loss;
    this->net_->Forward(bottom_vec, &loss);
    LOG(INFO) << "Iteration " << this->iter_ << ", loss = " << loss;
  }
  if (this->param_.test_interval() &&
      this->iter_ % this->param_.test_interval() == 0) {
    this->PlayAtari(0);
  }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void AtariSolver<Dtype>::PlayAtari(const int test_net_id) {
  LOG(INFO) << "Entering Game Playing Phase. Epsilon=" << epsilon_;
  Caffe::set_phase(Caffe::TEST);
  CHECK_NOTNULL(this->test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(this->net_.get());
  Net<Dtype>* test_net = this->test_nets_[test_net_id].get();
  ActionVect legal_actions = ale_.getLegalActionSet();
  const ALEScreen& screen = ale_.getScreen();
  vector<Datum> datum_vector(1);
  vector<Blob<Dtype>*> bottom_vec;
  const shared_ptr<MemoryDataLayer<Dtype> > memory_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<Dtype> >
      (this->test_nets_[test_net_id]->layers()[0]);
  CHECK(memory_layer) <<
      "Input Layer to the Atari Test Net must be a MemoryDataLayer.";
  Experience experience;
  leveldb::WriteBatch batch;
  int episode = 0;
  int experience_count = 0;
  int max_experiences = this->param_.test_iter(test_net_id);
  Action action_index;
  float f;
  while (experience_count < max_experiences) {
    int steps = 0;
    float totalReward = 0;
    while (!ale_.game_over()) {
      ReadScreenToDatum(screen, &(datum_vector[0]));
      ReadScreenToDatum(screen, experience.mutable_state());
      caffe_rng_uniform(1, 0.f, 1.f, &f);
      if (f < epsilon_) {
        action_index = Action(caffe_rng_rand() % legal_actions.size());
      } else {
        memory_layer->AddDatumVector(datum_vector);
        GetMaxAction(test_net->Forward(bottom_vec), &action_index);
      }
      float reward = ale_.act(legal_actions[action_index]);
      if (reward != Dtype(0)) {
        LOG(INFO) << "Reward " << reward;
      }
      totalReward += reward;
      steps++;

      experience.set_action(action_index);
      experience.set_reward(reward);
      ReadScreenToDatum(screen, experience.mutable_new_state());
      // {  // Display the network's predictions
      //   memory_layer->AddDatumVector(datum_vector);
      //   const vector<Blob<Dtype>*> output = test_net->Forward(bottom_vec);
      //   DisplayExperience(experience, *output[0]);
      // }

      // Add the experience to the experience memory
      experience_memory_.push_back(experience);
      if (experience_memory_.size() >= memory_size_) {
        experience_memory_.pop_front();
      }
      experience_count++;
    }
    LOG(INFO) << "Episode " << episode << " ended in " << steps
              << " steps with score " << totalReward;
    ale_.reset_game();
    episode++;
  }
  // Anneal epsilon
  epsilon_ = max(Dtype(0.1), epsilon_ - Dtype(experience_count / 1e6));
  LOG(INFO) << "Annealing Epsilon to " << epsilon_;
  Caffe::set_phase(Caffe::TRAIN);
  LOG(INFO) << "Leaving Game Playing Phase.";
}

template <typename Dtype>
void AtariSolver<Dtype>::DisplayScreen(const ALEScreen& screen) {
  int screen_height = screen.height();
  int screen_width = screen.width();
  unsigned char* pixels = screen.getArray();
  cv::Mat mat(screen_height, screen_width, CV_8UC4);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      cv::Vec4b& rgba = mat.at<cv::Vec4b>(i, j);
      int r,g,b;
      ale_.theOSystem->p_export_screen->
          get_rgb_from_palette(pixels[i*screen_width+j],r,g,b);
      rgba[0] = static_cast<unsigned char>(b); // Blue Channel
      rgba[1] = static_cast<unsigned char>(g); // Green Channel
      rgba[2] = static_cast<unsigned char>(r); // Red Channel
      rgba[3] = static_cast<unsigned char>(255); // Alpha Channel
    }
  }
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", mat);
  cv::waitKey(0);
  return;
}

template <typename Dtype>
void AtariSolver<Dtype>::DisplayScreenFromDatum(const Datum& datum) {
  const int screen_height = datum.height();
  const int screen_width = datum.width();
  const int screen_bytes = screen_height * screen_width;
  const string& data = datum.data();
  cv::Mat mat(screen_height, screen_width, CV_8UC4);
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      int offset = i * screen_width + j;
      cv::Vec4b& rgba = mat.at<cv::Vec4b>(i, j);
      rgba[2] = data[0 * screen_bytes + offset]; // Red Channel
      rgba[1] = data[1 * screen_bytes + offset]; // Green Channel
      rgba[0] = data[2 * screen_bytes + offset]; // Blue Channel
      rgba[3] = static_cast<unsigned char>(255); // Alpha Channel
    }
  }
  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", mat);
  cv::waitKey(0);
  return;
}

template <typename Dtype>
void AtariSolver<Dtype>::DisplayExperience(const Experience& experience,
                                           const Blob<Dtype>* activations) {
  const Datum& state = experience.state();
  const Datum& new_state = experience.new_state();
  CHECK_EQ(state.height(), new_state.height());
  CHECK_EQ(state.width(), new_state.width());
  CHECK_EQ(state.channels(), new_state.channels());
  const int screen_height = state.height();
  const int screen_width = state.width();
  const int screen_bytes = screen_height * screen_width;
  const int text_height = 100;
  const int display_height = screen_height + text_height;
  const int display_width = 2 * screen_width;
  const string& state_data = state.data();
  const string& new_state_data = new_state.data();
  cv::Mat mat(display_height, max(display_width, 400), CV_8UC4, cv::Scalar::all(0));
  for (int i = 0; i < screen_height; ++i) {
    for (int j = 0; j < display_width; ++j) {
      const string& data = j < screen_width ? state_data : new_state_data;
      int offset = i * screen_width + j % screen_width;
      cv::Vec4b& rgba = mat.at<cv::Vec4b>(i, j);
      rgba[2] = data[0 * screen_bytes + offset]; // Red Channel
      rgba[1] = data[1 * screen_bytes + offset]; // Green Channel
      rgba[0] = data[2 * screen_bytes + offset]; // Blue Channel
      rgba[3] = static_cast<unsigned char>(255); // Alpha Channel
    }
  }
  int offset = 15;
  stringstream ss;
  ss << "A: " << experience.action() << " R: " << experience.reward();
  cv::Point org(0, screen_height + offset);
  putText(mat, ss.str(), org, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255));
  offset += 15;

  if (activations != NULL) {
    ss.precision(3);
    Dtype print_data[activations->count()];
    caffe_gpu_memcpy(sizeof(Dtype)*activations->count(),
                     activations->gpu_data(), &print_data);
    Dtype max = print_data[0];
    int n = 0;
    while (n < activations->count()) {
      ss.str("");
      for (int i = 0; i < 5; ++i) {
        if (n >= activations->count()) {
          break;
        }
        if (print_data[n] > max) {
          max = print_data[n];
        }
        ss << std::fixed << print_data[n++] << " ";
      }
      org = cv::Point(0, screen_height + offset);
      offset += 15;
      putText(mat, ss.str(), org, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255));
    }
    ss.str("");
    ss << "Max Val: " << std::scientific << max;
    org = cv::Point(0, screen_height + offset);
    putText(mat, ss.str(), org, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar::all(255));
    offset += 15;
  }

  cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
  cv::imshow("Display window", mat);
  cv::waitKey(0);
  return;
}

template <typename Dtype>
void AtariSolver<Dtype>::ReadScreenToDatum(const ALEScreen& screen,
                                           Datum* datum) {
  CHECK_EQ(target_channels_, 3)
      << "Only 3 channel conversion is currently supported.";
  const int screen_height = screen.height();
  const int screen_width = screen.width();
  unsigned char* pixels = screen.getArray();

  // TODO(mhauskn): Convert screen to grayscale?
  const float y_scale = screen_height / float(target_height_);
  const float x_scale = screen_width / float(target_width_);
  const int screen_size = target_height_ * target_width_;
  const int screen_bytes = target_channels_ * screen_size;
  char str_buffer[screen_bytes];
  int r,g,b;
  int offset;
  for (int y = 0; y < target_height_; ++y) {
    for (int x = 0; x < target_width_; ++x) {
      offset = ((int) (y_scale * y)) * screen_width + ((int) (x_scale * x));
      ale_.theOSystem->p_export_screen->
          get_rgb_from_palette(pixels[offset],r,g,b);
      offset = y * target_width_ + x;
      str_buffer[offset] = (char) r;
      str_buffer[screen_size + offset] = (char) g;
      str_buffer[2 * screen_size + offset] = (char) b;
    }
  }
  datum->set_channels(target_channels_);
  datum->set_height(target_height_);
  datum->set_width(target_width_);
  datum->set_data(str_buffer, screen_bytes);
  return;
}

template <typename Dtype>
void AtariSolver<Dtype>::GetMaxAction(const vector<Blob<Dtype>*>& output_blobs,
                                      Action* max_actions,
                                      Dtype* max_action_vals) {
  int num_legal_actions = ale_.getLegalActionSet().size();
  Blob<Dtype>* output_blob = output_blobs[0];
  CHECK_GE(output_blob->channels(), num_legal_actions)
      << "Output layer has fewer channels than number of legal actions.";

  const int count = output_blob->count();
  Dtype* output_data;
  if (Caffe::mode() == Caffe::GPU) {
    output_data = new Dtype[count];
    caffe_gpu_memcpy(sizeof(Dtype) * count, output_blobs[0]->gpu_data(),
                     output_data);
  } else if (Caffe::mode() == Caffe::CPU) {
    output_data = output_blobs[0]->mutable_cpu_data();
  } else {
    LOG(FATAL) << "Unknown caffe mode.";
  }

  if (max_actions != NULL) {
    for (int n = 0; n < output_blob->num(); ++n) {
      int start_indx = output_blob->offset(n);
      Dtype max_val = output_data[start_indx];
      vector<Action> max_inds;
      max_inds.push_back(Action(0));
      for (int i = start_indx + 1; i < start_indx + num_legal_actions; ++i) {
        if (output_data[i] > max_val) {
          max_inds.clear();
          max_val = output_data[i];
          max_inds.push_back(Action(i - start_indx));
        } else if (output_data[i] == max_val) {
          LOG(INFO) << "Collision!";
          max_inds.push_back(Action(i - start_indx));
        }
      }
      Action max_action = max_inds[caffe_rng_rand() % max_inds.size()];
      max_actions[n] = max_action;
      if (max_action_vals != NULL) {
        max_action_vals[n] = max_val;
      }
    }
  } else if (max_action_vals != NULL) {
    // Quicker version to just get the maximum values
    for (int n = 0; n < output_blob->num(); ++n) {
      int start_indx = output_blob->offset(n);
      Dtype max_val = output_data[start_indx];
      for (int i = start_indx + 1; i < start_indx + num_legal_actions; ++i) {
        if (output_data[i] > max_val) {
          max_val = output_data[i];
        }
      }
      CHECK(!isnan(max_val));
      max_action_vals[n] = max_val;
    }
  } else {
    LOG(FATAL) << "Both max_actions and max_action_vals cannot be null!";
  }

  if (Caffe::mode() == Caffe::GPU) {
    delete[] output_data;
  }

  return;
}

template <typename Dtype>
Dtype AtariSolver<Dtype>::ForwardBackward(
    const vector<Blob<Dtype>*>& bottom_vec) {
  const shared_ptr<MemoryDataLayer<Dtype> > memory_layer =
      boost::dynamic_pointer_cast<MemoryDataLayer<Dtype> >
      (this->net_->layers()[0]);
  CHECK(memory_layer) <<
      "Input Layer to the Atari Test Net must be a MemoryDataLayer.";

  int batch_size = memory_layer->batch_size();
  vector<int> actions;
  vector<float> rewards;
  vector<Datum> state_datum_vector(batch_size);
  vector<Datum> new_state_datum_vector(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    Experience& exp = experience_memory_.at(
        caffe_rng_rand() % experience_memory_.size());
    actions.push_back(exp.action());
    rewards.push_back(exp.reward());
    state_datum_vector[i] = exp.state();
    new_state_datum_vector[i] = exp.new_state();
  }

  // Run the first forward pass on the next state values
  memory_layer->AddDatumVector(new_state_datum_vector);
  this->net_->Forward(bottom_vec);

  // Access the output values of the network.
  const vector<vector<Blob<Dtype>*> >& top_vecs = this->net_->top_vecs();
  // Here we assume that the last layer is a loss layer so the 2nd
  // to last layer contains the blobs of interest.
  const vector<Blob<Dtype>*>& output_blobs = top_vecs[top_vecs.size() - 2];
  // PrintBlob("NextStateOutputs", *output_blobs[0], false);
  // Compute the targets for forward-backward
  ComputeLabels(output_blobs, actions, rewards, gamma_,
                memory_layer->labels_ptr());
  // Run the next forward pass on the previous state values. This is
  // done automatically by the ExperienceDataLayer.
  memory_layer->AddDatumVector(state_datum_vector);
  this->net_->Forward(bottom_vec);
  // Get the loss layer to do special hacks on it...
  const shared_ptr<EuclideanLossLayer<Dtype> > loss_layer =
      boost::dynamic_pointer_cast<EuclideanLossLayer<Dtype> >
      (this->net_->layers().back());
  CHECK(loss_layer) <<
      "Last Layer of the Atari Test Net must be a Euclidean Loss Layer.";
  Blob<Dtype>* diff = loss_layer->mutable_diff();
  // PrintBlob("Pre-Zeroed Diff", *diff, false);
  // Update the diff blob from the loss layer to only take the diff of
  // the output nodes for which actions were taken.
  ClearNonActionDiffs(actions, diff);
  // PrintBlob("Post-Zeroed Diff", *diff, false);
  // Get the actual loss - Only penalize net for predictions of
  // actions that were actually taken.
  Dtype loss = GetEuclideanLoss(actions, diff);
  // Run the backwards pass on the network and return the loss.
  this->net_->Backward();
  return loss;
}

template <typename Dtype>
void AtariSolver<Dtype>::PrintBlob(string name, const Blob<Dtype>& blob,
                                   bool cpu) {
  cout << "Blob: " << name
       << " num=" << blob.num()
       << " channels=" << blob.channels()
       << " height=" << blob.height()
       << " width=" << blob.width()
       << " count=" << blob.count()
       << endl;
  Dtype print_data[blob.count()];
  const Dtype* data;
  if (cpu) {
    data = blob.cpu_data();
  } else {
    const Dtype* gpu_data = blob.gpu_data();
    caffe_gpu_memcpy(sizeof(Dtype)*blob.count(), gpu_data, &print_data);
    data = print_data;
  }
  for (int n = 0; n < blob.num(); ++n) {
    cout.precision(2);
    cout << "n=" << n << " ";
    int start_indx = blob.offset(n);
    for (int i = start_indx; i < start_indx + blob.channels(); ++i) {
      cout << data[i] << " ";
    }
    cout << endl;
  }
}

template <typename Dtype>
Dtype AtariSolver<Dtype>::GetEuclideanLoss(const vector<int>& actions,
                                           Blob<Dtype>* diff) {
  int num = diff->num();
  int channels = diff->channels();
  CHECK_EQ(num, actions.size()) << "Diff size must equal action size!";
  CHECK_EQ(diff->height(), 1) << "Diff has height other than 1!";
  CHECK_EQ(diff->width(), 1) << "Diff has width other than 1!";
  for (int i = 0; i < actions.size(); ++i) {
    CHECK_LT(actions[i], channels)
        << "Actions[" << i << "] has value " << actions[i]
        << " but we only have " << channels << " channels.";
  }

  Dtype loss(0);
  if (Caffe::mode() == Caffe::GPU) {
    Dtype tmp;
    const Dtype* gpu_data = diff->gpu_data();
    // PrintBlob("Diff", *diff, false);
    // cout << "Diff: ";
    // setprecision(3);
    for (int n = 0; n < num; ++n) {
      int chan = actions[n];
      caffe_gpu_memcpy(sizeof(Dtype), &gpu_data[n * channels + chan], &tmp);
      // cout << tmp << " ";
      loss += tmp * tmp;
    }
    // cout << endl;
  } else if (Caffe::mode() == Caffe::CPU) {
    const Dtype* diff_data = diff->cpu_data();
    for (int n = 0; n < num; ++n) {
      int chan = actions[n];
      Dtype tmp = diff_data[n * channels + chan];
      loss += tmp * tmp;
    }
  } else {
    LOG(FATAL) << "Unknown caffe mode.";
  }

  return loss / Dtype(2 * num);
}

template <typename Dtype>
void AtariSolver<Dtype>::ClearNonActionDiffs(const vector<int>& actions,
                                             Blob<Dtype>* diff) {
  int num = diff->num();
  int channels = diff->channels();
  CHECK_EQ(num, actions.size()) << "Diff size must equal action size!";
  CHECK_EQ(diff->height(), 1) << "Diff has height other than 1!";
  CHECK_EQ(diff->width(), 1) << "Diff has width other than 1!";
  for (int i = 0; i < actions.size(); ++i) {
    CHECK_LT(actions[i], channels)
        << "actions[" << i << "] has value " << actions[i]
        << " but we only have " << channels << " channels.";
  }

  // Zero all the diffs except those corresponding to actions.
  if (Caffe::mode() == Caffe::GPU) {
    Dtype* diff_data = diff->mutable_gpu_data();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        if (actions[n] != c) {
          caffe_gpu_set(1, Dtype(0.), &diff_data[n * channels + c]);
        }
      }
    }
  } else if (Caffe::mode() == Caffe::CPU) {
    Dtype* diff_data = diff->mutable_cpu_data();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        if (actions[n] != c) {
          diff_data[n * channels + c] = Dtype(0);
        }
      }
    }
  } else {
    LOG(FATAL) << "Unknown caffe mode.";
  }
  return;
}

template <typename Dtype>
void AtariSolver<Dtype>::ComputeLabels(const vector<Blob<Dtype>*>& output_blobs,
                                       const vector<int>& actions,
                                       const vector<float>& rewards,
                                       const Dtype gamma,
                                       Blob<Dtype>* labels) {
  // LOG(INFO) << "First Forward Pass Output:";
  // PrintBlob("Output", *output_blobs[0], false);
  int batch_size = output_blobs[0]->num();
  CHECK_EQ(labels->count(), output_blobs[0]->count())
      << "Labels count does not equal output_blobs[0] count.";
  CHECK_EQ(batch_size, rewards.size()) << "Output size must equal reward size!";
  CHECK_EQ(batch_size, actions.size()) << "Output size must equal action size!";

  Dtype* label_data = labels->mutable_cpu_data();

  // Zero all the labels
  for (int i = 0; i < labels->count(); ++i) {
    label_data[i] = Dtype(0);
  }

  // Compute the max over all next-state values
  Dtype max_action_vals[batch_size];
  GetMaxAction(output_blobs, NULL, max_action_vals);

  // Compute the gamma-discounted max over next state actions and
  // use this compute the labels.
  for (int n = 0; n < batch_size; ++n) {
    Dtype reward(rewards[n]);
    if (rescale_reward_) {
      if (reward > 0) {
        reward = Dtype(1.0);
      } else if (reward < 0) {
        reward = Dtype(-1.0);
      }
    }
    Dtype target(reward + gamma * max_action_vals[n]);
    int offset = labels->offset(n) + actions[n];
    label_data[offset] = target;
  }
  // PrintBlob("Labels", *labels, true);
  return;
}

INSTANTIATE_CLASS(Solver);
INSTANTIATE_CLASS(SGDSolver);
INSTANTIATE_CLASS(NesterovSolver);
INSTANTIATE_CLASS(AdaGradSolver);
INSTANTIATE_CLASS(AtariSolver);

}  // namespace caffe
