#!/usr/bin/env bash

# add gem and bundle mirror.
gem sources --add https://mirrors.tuna.tsinghua.edu.cn/rubygems/ --remove https://rubygems.org/
bundle config mirror.https://rubygems.org https://mirrors.tuna.tsinghua.edu.cn/rubygems

if [ -f package.json ]; then
  bash -i -c "nvm install --lts && nvm install-latest-npm"
  npm i
  npm run build
fi

echo "create complete."
