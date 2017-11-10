if [ -d script ]
then
    cd script
    rm -rf ../data/cache/*
    cd ..
else
    rm -rf ../data/cache/*
fi
