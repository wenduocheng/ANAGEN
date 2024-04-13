wget http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz
tar -xzf deepsea_train_bundle.v0.9.tar.gz
mv deepsea_train/train.mat deepsea_train/valid.mat deepsea_train/test.mat .
rm -r deepsea_train