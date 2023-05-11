all:  main

CUPTI_PATH=/usr/local/cuda/extras/CUPTI
INCLUDES = -I ../include -I /usr/local/cuda/extras/CUPTI/include 


CXXARGS = -std=c++11 -g
CXXARGS += -Xcompiler -DNDEBUG
CXXARGS += -Xptxas="-dlcm=cg"  # --def-load-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).
CXXARGS += -Xptxas="-dscm=cg"  # --def-store-cache  "cg" -> Cache at global level (cache in L2 and below, not L1).

LIBS = -lcuda -L$(CUPTI_PATH)/lib64 -lcupti



main: main.cu	
	nvcc $(CXXARGS) $(INCLUDES) $(LIBS) main.cu -o main



clean: 
	rm -f *.o main
