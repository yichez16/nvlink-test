all:  main

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include 


CXXARGS = -std=c++14 -g
CXXARGS += -Xcompiler -DNDEBUG
LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti



main: main.cu	
	# nvcc $(CXXARGS) $(INCLUDES) $(LIBS) main.cu -o main
	nvcc -std=c++17 -o main-l1 main.cu -O3 -Xptxas="-dlcm=ca"
	
    nvcc -std=c++17 -o main-l2 main.cu -O3 -Xptxas="-dlcm=cg"


clean: 
	rm -f *.o main
