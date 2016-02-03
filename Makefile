NVCC=nvcc
CC=gcc

VPATH=cu
TARGET=lcuda

INCS=-I/usr/local/cuda/include -I$(PWD)

CFLAGS+=$(INCS) -fPIC

C_OBJS=lcudafloat.o lcudamorph.o lcudatypes.o
CU_OBJS=lcudamorphfloat.o morphology.o

all: lcuda

%.o: %.cu
	$(NVCC) -c $(INCS) -Xcompiler -fPIC --use_fast_math -arch sm_52 -o $@ $<

lcuda: $(C_OBJS) $(CU_OBJS)
	$(NVCC) -shared -o lib$(TARGET).so $(C_OBJS) $(CU_OBJS)

clean:
	@rm -f *.o
	@rm -f lib$(TARGET).so
