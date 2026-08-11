NVCC_FLAGS = -std=c++17 -O3 -DNDEBUG -w
NVCC_LDFLAGS = -lcublas -lcuda
OUT_DIR = out
PROFILE_DIR = profile

CUDA_OUTPUT_FILE = -o $(OUT_DIR)/$@
NCU_PATH := $(shell which ncu)
NCU_COMMAND = $(NCU_PATH) --set full --import-source yes

NVCC_FLAGS += --expt-relaxed-constexpr --expt-extended-lambda --use_fast_math -Xcompiler=-fPIE -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
NVCC_FLAGS += -arch=sm_90a

NVCC_BASE = nvcc $(NVCC_FLAGS) $(NVCC_LDFLAGS) -lineinfo

ld_matrix_x1: ld_matrix_x1.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

ld_matrix_x2: ld_matrix_x2.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

ld_matrix_x4: ld_matrix_x4.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

st_matrix_x1: st_matrix_x1.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

st_matrix_x2: st_matrix_x2.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

st_matrix_x4: st_matrix_x4.cu 
	$(NVCC_BASE) $^ $(CUDA_OUTPUT_FILE)

compile_all: 
	make ld_matrix_x1
	make ld_matrix_x2
	make ld_matrix_x4
	make st_matrix_x1
	make st_matrix_x2
	make st_matrix_x4

run_all: compile_all
	compute-sanitizer ./$(OUT_DIR)/ld_matrix_x1
	compute-sanitizer ./$(OUT_DIR)/ld_matrix_x2
	compute-sanitizer ./$(OUT_DIR)/ld_matrix_x4
	compute-sanitizer ./$(OUT_DIR)/st_matrix_x1
	compute-sanitizer ./$(OUT_DIR)/st_matrix_x2
	compute-sanitizer ./$(OUT_DIR)/st_matrix_x4

clean:
	rm $(OUT_DIR)/*