mkdir -p snapshots snapshots/lores snapshots/midres snapshots/hires

# 3m0s to get to ~0.00013
rm snapshots/lores/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 64 -T 160 -d 4 -o weights_lores.txt -S -b snapshots/lores/out

# Note we do full updates more often, train/replace 2 models at a time, fuzz a bit more
# 6m3s to get to 0.00035
#rm snapshots/midres/*.png; time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 96 -T 64 -d 2 -i weights_lores.txt -o weights_midres.txt -S -b snapshots/midres/out

#time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 128 -F 32 -R 0.003 -f 15000 -a 0.1 -i weights_midres.txt -o weights_hiresA.txt -S -b hiresA
