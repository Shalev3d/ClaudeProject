# Quartus TCL script to create DE10-Lite Matrix Multiplier project
# Run with: quartus_sh -t create_project.tcl

# Close any open project first
catch {project_close}

# Create new project (overwrite if exists)
project_new de10_lite_matrix_multiplier -overwrite

# Set device
set_global_assignment -name FAMILY "MAX 10"
set_global_assignment -name DEVICE 10M50DAF484C7G
set_global_assignment -name TOP_LEVEL_ENTITY de10_lite_top
set_global_assignment -name ORIGINAL_QUARTUS_VERSION 24.1
set_global_assignment -name PROJECT_CREATION_TIME_DATE [clock format [clock seconds]]

# Add source files
set_global_assignment -name SYSTEMVERILOG_FILE de10_lite_top.sv
set_global_assignment -name SYSTEMVERILOG_FILE matrix_multiplier.sv
set_global_assignment -name SYSTEMVERILOG_FILE uart_controller.sv
set_global_assignment -name SYSTEMVERILOG_FILE host_interface.sv
set_global_assignment -name SYSTEMVERILOG_FILE pll_100mhz.sv

# Add constraints
set_global_assignment -name SDC_FILE de10_lite_constraints.sdc

# Pin assignments for DE10-Lite
set_location_assignment PIN_P11 -to MAX10_CLK1_50
set_location_assignment PIN_N14 -to MAX10_CLK2_50

# Reset button (active low)
set_location_assignment PIN_B8 -to KEY[0]
set_location_assignment PIN_A7 -to KEY[1]

# Switches
set_location_assignment PIN_C10 -to SW[0]
set_location_assignment PIN_C11 -to SW[1]
set_location_assignment PIN_D12 -to SW[2]
set_location_assignment PIN_C12 -to SW[3]
set_location_assignment PIN_A12 -to SW[4]
set_location_assignment PIN_B12 -to SW[5]
set_location_assignment PIN_A13 -to SW[6]
set_location_assignment PIN_A14 -to SW[7]
set_location_assignment PIN_B14 -to SW[8]
set_location_assignment PIN_F15 -to SW[9]

# LEDs
set_location_assignment PIN_A8 -to LEDR[0]
set_location_assignment PIN_A9 -to LEDR[1]
set_location_assignment PIN_A10 -to LEDR[2]
set_location_assignment PIN_B10 -to LEDR[3]
set_location_assignment PIN_D13 -to LEDR[4]
set_location_assignment PIN_C13 -to LEDR[5]
set_location_assignment PIN_E14 -to LEDR[6]
set_location_assignment PIN_D14 -to LEDR[7]
set_location_assignment PIN_A11 -to LEDR[8]
set_location_assignment PIN_B11 -to LEDR[9]

# 7-segment displays
set_location_assignment PIN_C14 -to HEX0[0]
set_location_assignment PIN_E15 -to HEX0[1]
set_location_assignment PIN_C15 -to HEX0[2]
set_location_assignment PIN_C16 -to HEX0[3]
set_location_assignment PIN_E16 -to HEX0[4]
set_location_assignment PIN_D17 -to HEX0[5]
set_location_assignment PIN_C17 -to HEX0[6]
set_location_assignment PIN_D15 -to HEX0[7]

set_location_assignment PIN_C18 -to HEX1[0]
set_location_assignment PIN_D18 -to HEX1[1]
set_location_assignment PIN_E18 -to HEX1[2]
set_location_assignment PIN_B16 -to HEX1[3]
set_location_assignment PIN_A17 -to HEX1[4]
set_location_assignment PIN_A18 -to HEX1[5]
set_location_assignment PIN_B17 -to HEX1[6]
set_location_assignment PIN_A16 -to HEX1[7]

set_location_assignment PIN_B20 -to HEX2[0]
set_location_assignment PIN_A20 -to HEX2[1]
set_location_assignment PIN_B19 -to HEX2[2]
set_location_assignment PIN_A21 -to HEX2[3]
set_location_assignment PIN_B21 -to HEX2[4]
set_location_assignment PIN_C22 -to HEX2[5]
set_location_assignment PIN_B22 -to HEX2[6]
set_location_assignment PIN_A19 -to HEX2[7]

set_location_assignment PIN_F21 -to HEX3[0]
set_location_assignment PIN_E22 -to HEX3[1]
set_location_assignment PIN_E21 -to HEX3[2]
set_location_assignment PIN_C19 -to HEX3[3]
set_location_assignment PIN_C20 -to HEX3[4]
set_location_assignment PIN_D19 -to HEX3[5]
set_location_assignment PIN_E17 -to HEX3[6]
set_location_assignment PIN_D22 -to HEX3[7]

set_location_assignment PIN_F18 -to HEX4[0]
set_location_assignment PIN_E20 -to HEX4[1]
set_location_assignment PIN_E19 -to HEX4[2]
set_location_assignment PIN_J18 -to HEX4[3]
set_location_assignment PIN_H19 -to HEX4[4]
set_location_assignment PIN_F19 -to HEX4[5]
set_location_assignment PIN_F20 -to HEX4[6]
set_location_assignment PIN_F17 -to HEX4[7]

set_location_assignment PIN_J20 -to HEX5[0]
set_location_assignment PIN_K20 -to HEX5[1]
set_location_assignment PIN_L18 -to HEX5[2]
set_location_assignment PIN_N18 -to HEX5[3]
set_location_assignment PIN_M20 -to HEX5[4]
set_location_assignment PIN_N19 -to HEX5[5]
set_location_assignment PIN_N20 -to HEX5[6]
set_location_assignment PIN_L19 -to HEX5[7]

# UART (GPIO pins - adjust based on your connection)
set_location_assignment PIN_V10 -to UART_RXD
set_location_assignment PIN_W10 -to UART_TXD

# SDRAM (optional)
set_location_assignment PIN_U17 -to DRAM_CLK
set_location_assignment PIN_W19 -to DRAM_CKE
set_location_assignment PIN_V18 -to DRAM_ADDR[0]
set_location_assignment PIN_V17 -to DRAM_ADDR[1]
set_location_assignment PIN_U18 -to DRAM_ADDR[2]
set_location_assignment PIN_U17 -to DRAM_ADDR[3]
set_location_assignment PIN_T18 -to DRAM_ADDR[4]
set_location_assignment PIN_T17 -to DRAM_ADDR[5]
set_location_assignment PIN_T19 -to DRAM_ADDR[6]
set_location_assignment PIN_R18 -to DRAM_ADDR[7]
set_location_assignment PIN_P18 -to DRAM_ADDR[8]
set_location_assignment PIN_P19 -to DRAM_ADDR[9]
set_location_assignment PIN_T20 -to DRAM_ADDR[10]
set_location_assignment PIN_P20 -to DRAM_ADDR[11]
set_location_assignment PIN_R20 -to DRAM_ADDR[12]

# Set I/O standards
set_instance_assignment -name IO_STANDARD "3.3-V LVTTL" -to *

# Compilation settings
set_global_assignment -name PARTITION_NETLIST_TYPE SOURCE -section_id Top
set_global_assignment -name PARTITION_FITTER_PRESERVATION_LEVEL PLACEMENT_AND_ROUTING -section_id Top
set_global_assignment -name PARTITION_COLOR 16764057 -section_id Top

# Close project
project_close

puts "Project created successfully!"