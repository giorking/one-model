enc=$1
echo "enc flag $enc"
rm -rf dist
git pull -p
mkdir dist && cd dist
git clone --branch support-text-embdings --depth 1 git@git.xiaojukeji.com:qiudanwang_i/segment-anything.git
git clone --branch main --depth 1 git@git.xiaojukeji.com:qiudanwang_i/one-model.git
if [ "$enc" = "true" ]; then
    echo "enc one_model"
    cd one_model && pyarmor gen -r --exclude "*.pyc" --exclude "*.pyo" --output . . && rm -rf .git docker start_docker.sh docs tests
else
    echo "..."
    cd one_model && rm -rf .git docker start_docker.sh docs
fi
