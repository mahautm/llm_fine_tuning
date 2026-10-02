#!/usr/bin/env bash

paramem_activate_env() {
  local load_cuda="${1:-1}"
  source ~/.bashrc
  if [ "$load_cuda" = "1" ]; then
    module load CUDA/12.1.0
  fi
  conda activate paramem
}
