# !/bin/bash

cargo run --release
cd anim/
./animate.sh
cd ../
rsixel final_lat.png
cp final_lat.png report/lines/

