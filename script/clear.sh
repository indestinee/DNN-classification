if [ -d script ]
then
    cd script
    rm -rf ../data/cache
    rm -rf ../train_log
    rm -rf ../result
    cd ..
else
    rm -rf ../data/cache
    rm -rf ../train_log
    rm -rf ../result
fi
