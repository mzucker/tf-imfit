mkdir -p results results/lores results/midres results/hires results/final

# 2m14s to get to 0.00013
rm -f results/lores/*.png && \
time python tf-imfit.py images/zz_rect.png -w images/zz_rect_weights.png \
     -s 64 -T 256 -a 0.01 \
     -o results/weights_lores.txt \
     -p 512 -S -x results/lores/out

# 7m18s to get to 0.00024
rm -f results/midres/*.png && \
time python tf-imfit.py images/zz_rect.png -w images/zz_rect_weights.png \
     -s 96 -T 384 -a 0.04 \
     -i results/weights_lores.txt -o results/weights_midres.txt \
     -p 512 -S -x results/midres/out

# 14m31s to 0.00036
rm -f results/hires/*png && \
time python tf-imfit.py images/zz_rect.png -w images/zz_rect_weights.png \
     -s 128 -T 512 -a0.02 -R 0.0002 \
     -i results/weights_midres.txt -o results/weights_hires.txt \
     -p 512 -S -x results/hires/out

# 1m54s to 0.00083
rm -f results/final/*png && \
time python tf-imfit.py images/zz_rect.png -w images/zz_rect_weights.png \
     -s 256 -t 0:00:01 -R 0.0002 \
     -i results/weights_hires.txt -o results/weights_final.txt \
     -p 512 -S -x results/final/out
