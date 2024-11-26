TARGETS = heat_t2
NVCCFLAGS = -O2 -I$(CUDAPATH)/include
NVCC = /usr/bin/nvcc
CUDAPATH = /usr/bin
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

all: $(TARGETS)

$(TARGETS): heat_t2.o
	$(NVCC) $(NVCCFLAGS) -o $@ $(LFLAGS) $^

heat_t2.o: heat_t2.c
	$(NVCC) $(NVCCFLAGS) -c heat_t2.c

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean

