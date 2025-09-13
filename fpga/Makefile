# Makefile for DE10-Lite Matrix Multiplier FPGA Build
# Provides easy command-line compilation and programming

# Project settings
PROJECT = de10_lite_matrix_multiplier
FAMILY = "MAX 10"
DEVICE = 10M50DAF484C7G

# Quartus tools
QUARTUS_SH = quartus_sh
QUARTUS_MAP = quartus_map
QUARTUS_FIT = quartus_fit
QUARTUS_ASM = quartus_asm
QUARTUS_STA = quartus_sta
QUARTUS_PGM = quartus_pgm

# Source files
VERILOG_SOURCES = de10_lite_top.sv matrix_multiplier.sv uart_controller.sv host_interface.sv
CONSTRAINT_FILES = de10_lite_constraints.sdc

# Output directory
OUTPUT_DIR = output_files

.PHONY: all clean create compile program help check-tools

# Default target
all: compile

# Check if Quartus tools are available
check-tools:
	@echo "Checking Quartus installation..."
	@command -v $(QUARTUS_SH) >/dev/null 2>&1 || { echo "Error: Quartus not found in PATH. Please install Quartus Prime Lite and add to PATH."; exit 1; }
	@echo "Quartus tools found ✓"

# Create project
create: check-tools
	@echo "Creating Quartus project..."
	$(QUARTUS_SH) -t create_project.tcl
	@echo "Project created successfully!"

# Compile the design
compile: create
	@echo "Starting FPGA compilation..."
	@echo "This may take 5-15 minutes depending on your system..."
	$(QUARTUS_SH) -t compile.tcl
	@echo "Compilation complete!"
	@echo "Programming file: $(OUTPUT_DIR)/$(PROJECT).sof"

# Program the FPGA
program: 
	@echo "Programming FPGA..."
	@if [ ! -f "$(OUTPUT_DIR)/$(PROJECT).sof" ]; then \
		echo "Error: Programming file not found. Run 'make compile' first."; \
		exit 1; \
	fi
	$(QUARTUS_SH) -t program.tcl
	@echo "FPGA programmed successfully!"

# Quick program using quartus_pgm directly
program-direct:
	@echo "Programming FPGA directly..."
	@if [ ! -f "$(OUTPUT_DIR)/$(PROJECT).sof" ]; then \
		echo "Error: Programming file not found. Run 'make compile' first."; \
		exit 1; \
	fi
	$(QUARTUS_PGM) -c "USB-Blaster" -m jtag -o "p;$(OUTPUT_DIR)/$(PROJECT).sof@1"

# List available programming cables
list-cables:
	@echo "Available programming cables:"
	$(QUARTUS_PGM) --list

# Clean build files
clean:
	@echo "Cleaning build files..."
	rm -rf $(OUTPUT_DIR)
	rm -rf db/
	rm -rf incremental_db/
	rm -f *.qws *.qpf *.qsf *.rpt *.summary *.done *.jdi *.pin *.sld
	@echo "Clean complete!"

# Analysis only (fast syntax check)
analyze: create
	@echo "Running analysis & synthesis only..."
	$(QUARTUS_MAP) $(PROJECT)
	@echo "Analysis complete!"

# Timing analysis
timing:
	@echo "Running timing analysis..."
	$(QUARTUS_STA) $(PROJECT)
	@echo "Check timing reports in $(OUTPUT_DIR)/"

# Resource utilization report
resources:
	@echo "Resource utilization:"
	@if [ -f "$(OUTPUT_DIR)/$(PROJECT).fit.rpt" ]; then \
		grep -A 20 "Total logic elements" $(OUTPUT_DIR)/$(PROJECT).fit.rpt || echo "Fit report not found"; \
	else \
		echo "No fit report found. Run 'make compile' first."; \
	fi

# Help
help:
	@echo "DE10-Lite Matrix Multiplier Build System"
	@echo "========================================"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Compile the complete design (default)"
	@echo "  create       - Create Quartus project only"
	@echo "  compile      - Full compilation (synthesis + place&route)"
	@echo "  program      - Program FPGA using TCL script"
	@echo "  program-direct - Program FPGA directly with quartus_pgm"
	@echo "  analyze      - Syntax check and analysis only (fast)"
	@echo "  timing       - Run timing analysis"
	@echo "  resources    - Show resource utilization"
	@echo "  list-cables  - List available programming cables"
	@echo "  clean        - Remove all build files"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - Quartus Prime Lite installed and in PATH"
	@echo "  - DE10-Lite connected via USB-Blaster"
	@echo "  - All source files present in current directory"
	@echo ""
	@echo "Example usage:"
	@echo "  make compile    # Build the design"
	@echo "  make program    # Program the FPGA"
	@echo "  make clean      # Clean build files"

# Quick build and program
flash: compile program
	@echo "Build and program complete!"

# Development cycle (quick recompile for small changes)
dev: analyze program-direct
	@echo "Quick development cycle complete!"