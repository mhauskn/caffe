#!/usr/bin/env sh

./build/tools/caffe train --solver=examples/mnist/frankenstein_solver.prototxt --weights=examples/mnist/frankenstein.caffemodel
