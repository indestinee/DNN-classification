if [ -d script ]
then
    cd script
    rm -rf ../data/cache/*
    rm -rf ../train_log/*
    cd ..
else
    rm -rf ../data/cache/*
    rm -rf ../train_log/*
fi
