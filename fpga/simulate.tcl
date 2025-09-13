# Quartus TCL script for simulation setup
# Run with: quartus_sh -t simulate.tcl

# Open the project
project_open de10_lite_matrix_multiplier

# Create simulation files
execute_module -tool eda -args "--simulation --tool=modelsim_oem --format=verilog"

# Set up simulation
set_global_assignment -name EDA_SIMULATION_TOOL "ModelSim-Altera (Verilog)"
set_global_assignment -name EDA_TIME_SCALE "1 ps" -section_id eda_simulation
set_global_assignment -name EDA_OUTPUT_DATA_FORMAT "VERILOG HDL" -section_id eda_simulation
set_global_assignment -name EDA_GENERATE_FUNCTIONAL_NETLIST OFF -section_id eda_simulation

# Generate simulation files
execute_module -tool eda

puts "Simulation files generated in simulation/modelsim directory"
project_close