docker rm -f one_model
commit_id=$(git rev-parse HEAD)
length=7
short_commit_id=$(echo "$commit_id" | rev | cut -c 1-$length | rev)

image_id=one_model:1.0_$short_commit_id
echo "start image id $image_id"

docker run --privileged=true --restart=always --name one_model -e HTTP_PORT=8800 -d --runtime=nvidia --gpus all --shm-size 10G -v /opt/product/license:/opt/product/license -v /opt/product/test_datas:/opt/product/test_datas -v /root/.cache:/root/.cache -v /usr/sbin/dmidecode:/usr/sbin/dmidecode -v /dev/mem:/dev/mem -v /tmp:/tmp -v /var/log:/var/log -v /var/run/docker.sock:/var/run/docker.sock --network host $image_id sh /opt/product/solution_backend/start.sh fg
