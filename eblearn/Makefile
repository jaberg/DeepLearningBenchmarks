
all: mnist_example_ipp.x mnist_example_noipp.x convnet_noipp.x convnet_ipp.x convnet96_ipp.x convnet96_noipp.x convnet256_ipp.x convnet256_noipp.x

clean:
	rm *.x

mnist_example_ipp.x : mnist_example.cc
	g++ -I${PUB_PREFIX}/eblearn_ipp -o mnist_example_ipp.x mnist_example.cc\
	    -L/u/bergstrj/pub/intel/ipp/6.1.2.051/em64t/sharedlib\
	    -L${PUB_PREFIX}/eblearn_ipp -leblearn -lippiem64t -pthread


mnist_example_noipp.x : mnist_example.cc
	g++ -O2 -I${PUB_PREFIX}/eblearn_noipp -o mnist_example_noipp.x mnist_example.cc\
	    -L${PUB_PREFIX}/eblearn_noipp -leblearn

convnet_noipp.x : convnet.cc
	g++ -O2 -I${PUB_PREFIX}/eblearn_noipp -o convnet_noipp.x convnet.cc\
	    -L${PUB_PREFIX}/eblearn_noipp -leblearn

convnet96_noipp.x : convnet96.cc
	g++ -O2 -I${PUB_PREFIX}/eblearn_noipp -o convnet96_noipp.x convnet96.cc\
	    -L${PUB_PREFIX}/eblearn_noipp -leblearn

convnet256_noipp.x : convnet256.cc
	g++ -O2 -I${PUB_PREFIX}/eblearn_noipp -o convnet256_noipp.x convnet256.cc\
	    -L${PUB_PREFIX}/eblearn_noipp -leblearn

convnet_ipp.x : convnet.cc
	g++ -DUSED_IPP -O2 -I${PUB_PREFIX}/eblearn_ipp -o convnet_ipp.x convnet.cc\
	    -L/u/bergstrj/pub/intel/ipp/6.1.2.051/em64t/sharedlib\
	    -L${PUB_PREFIX}/eblearn_ipp -leblearn -lippiem64t -pthread

convnet96_ipp.x : convnet96.cc
	g++ -DUSED_IPP -O2 -I${PUB_PREFIX}/eblearn_ipp -o convnet96_ipp.x convnet96.cc\
	    -L/u/bergstrj/pub/intel/ipp/6.1.2.051/em64t/sharedlib\
	    -L${PUB_PREFIX}/eblearn_ipp -leblearn -lippiem64t -pthread

convnet256_ipp.x : convnet256.cc
	g++ -DUSED_IPP -O2 -I${PUB_PREFIX}/eblearn_ipp -o convnet256_ipp.x convnet256.cc\
	    -L/u/bergstrj/pub/intel/ipp/6.1.2.051/em64t/sharedlib\
	    -L${PUB_PREFIX}/eblearn_ipp -leblearn -lippiem64t -pthread

