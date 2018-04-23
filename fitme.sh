# 1m20s to get to 0.00024
#time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 64 -T 256 -o weights_lores.txt -S -b lores

# Note we do full updates more often, train/replace 2 models at a time, fuzz a bit more
# 6m3s to get to 0.0035
#time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 96 -T 128 -F 16 -f 15000 -R 0.005 -a 0.3 -i weights_lores.txt -o weights_midres.txt -S -b midres

time python imfit.py images/zz_rect.png -w images/zz_rect_weights.png -s 128 -F 32 -R 0.003 -f 15000 -a 0.1 -i weights_midres.txt -o weights_hiresA.txt -S -b hiresA





