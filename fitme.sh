mkdir -p snapshots snapshots/lores snapshots/midres snapshots/hires

# 2m15s to get to ~0.00009
rm snapshots/lores/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 64 -T 256 -c 0.5 -P 0.01 -F32 -o weights_lores.txt -S -x snapshots/lores/out

# Note we do full updates more often, train/replace 2 models at a time, fuzz a bit more
# 7m47s to get to 0.000293
#rm snapshots/midres/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 96 -T 256 -c0.1 -P0.1 -b20 -F16 -i weights_lores.txt -o weights_midres.txt -S -x snapshots/midres/out

#rm snapshots/hires/*png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 128 -T 256 -d 2 -i weights_midres.txt -o weights_hires.txt -S -x snapshots/hires/out

