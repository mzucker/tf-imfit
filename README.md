# tf-imfit

TensorFlow-based rewrite of <https://github.com/mzucker/imfit>

You will need:

  - TensorFlow (I used v1.3 and 1.7, should work with versions in between as well)
  - PIL or Pillow (for Image loading/saving)
 
To do a full fit, just run [`fitme.sh`](fitme.sh). See that file for
example runtimes and objective function values collected with my
NVidia GTX 1080 GPU. Your runtimes and objective function values may
vary due to hardware differences.

If you just want to check the accuracy of an existing fit or visualize
its output, you can run something like:

    python ./tf-imfit.py -w images/zz_rect_weights.png images/zz_rect.png \
           -s256 -i params/zz_rect_weighted_256px.txt -p512 -T 0
    
Although you will want a fancy GPU to do the error minimization,
visualization/verification like the command above should run in
reasonable time on a CPU.
