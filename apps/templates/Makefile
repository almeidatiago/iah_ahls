VPP := $(XILINX_VITIS)/bin/v++
EMCONFIGUTIL := $(XILINX_VITIS)/bin/emconfigutil
TARGET:= hw
DEVICE:=xilinx_u55c_gen3x16_xdma_3_202210_1
#DEVICE:= xilinx_u250_gen3x16_xdma_4_1_202210_1
#DEVICE:= xilinx_u200_gen3x16_xdma_2_202110_1

# kernel targets
KRNL_XO0 := ./version_$(version)/kernel_"template".$(TARGET).xo
#KRNL_XO1 := kernel_approx.$(TARGET).xo
XCLBIN := ./version_$(version)/kernel.$(TARGET).xclbin

# host target
HOST_EXE := host.exe

# config files target
EMCONFIG_FILE := emconfig.json

VPP_OPTS := -s -t $(TARGET) --platform $(DEVICE)
VPP_OPTS += --temp_dir ./version_$(version)/all_builds/$(TARGET)
VPP_OPTS += --report_dir ./version_$(version)/all_logs/$(TARGET)
VPP_OPTS += --log_dir ./version_$(version)/all_logs/$(TARGET)
#VPP_OPTS += --profile.exec all:all
#VPP_OPTS += --profile.data all:all:all

CFLAGS := -g -std=c++1y -I$(XILINX_XRT)/include 
LFLAGS := -L$(XILINX_XRT)/lib -lOpenCL -lxrt_coreutil -lrt -luuid

#Include Required Host Source Files
#CFLAGS += -I../common/includes/xcl2
HOST_SRCS += ./host/host.cpp 
# Host compiler global settings

# run time args
CMD_ARGS:= kernel.${TARGET}.xclbin

# primary build targets
.PHONY: xclbin host all

xo:  $(KRNL_XO0)
xclbin:  $(XCLBIN)
host: $(HOST_EXE) 

all: xclbin host 

clean:
	-$(RM) $(EMCONFIG_FILE) $(HOST_EXE) $(XCLBIN) 

cleanall: clean	
	rm -rf all_* *.log *.xclbin *.compile_summary *.link_summary *.info *.xo

# kernel rules
$(KRNL_XO0): ./src/"template"_$(version).cpp
	$(RM) $@
	$(VPP) $(VPP_OPTS) -c -k "template"_$(version) -I'$(<D)' -o $@ $+
	
#$(KRNL_XO1): ./src/sobel_approx.cpp
#	$(RM) $@
#	$(VPP) $(VPP_OPTS) -c -k sobel_approx -I'$(<D)' -o $@ $+

$(XCLBIN): $(KRNL_XO0) 
	$(VPP) $(VPP_OPTS) -l -o $@ $(KRNL_XO0) $+ 

# host rules
$(HOST_EXE): $(HOST_SRCS) 
	g++ $(CFLAGS) -o $@ $+ $(LFLAGS)
	@echo 'Compiled Host Executable: $(HOST_EXE)'

$(EMCONFIG_FILE):
	$(EMCONFIGUTIL) --nd 1 --od . --platform $(DEVICE)

run: $(XCLBIN) $(HOST_EXE) $(EMCONFIG_FILE)
	XCL_EMULATION_MODE=${TARGET} ./$(HOST_EXE) $(CMD_ARGS)
