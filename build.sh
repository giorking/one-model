#!/bin/bash
set -e

# enc=true
# TODO 发布之前需要设置代码加密
enc=false
sh clone.sh $enc

commit_id=$(git rev-parse HEAD)
length=7
short_commit_id=$(echo "$commit_id" | rev | cut -c 1-$length | rev)

docker build --network host -t one_model:1.0_$short_commit_id -f docker/Dockerfile .

echo "build done"

rm -rf dist
