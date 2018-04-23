mkdir -p snapshots snapshots/lores snapshots/midres snapshots/hires

# 2m50s to get to ~0.00013
rm snapshots/lores/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 64 -T 192 -c 0.2 -P 0.1 -o weights_lores.txt -S -b snapshots/lores/out

# Note we do full updates more often, train/replace 2 models at a time, fuzz a bit more
# 7m47s to get to 0.000293
#rm snapshots/midres/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 96 -T 32 -d 2 -i weights_lores.txt -o weights_midres.txt -S -b snapshots/midres/out

#rm snapshots/hires/*png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 128 -T 256 -d 2 -i weights_midres.txt -o weights_hires.txt -S -b snapshots/hires/out

