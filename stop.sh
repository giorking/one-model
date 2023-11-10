ps -ef | grep one_model | grep -v grep | awk '{print $2}' | xargs kill -9
rm -rf nohup.out
