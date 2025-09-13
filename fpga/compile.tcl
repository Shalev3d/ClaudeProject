# Quartus TCL script to compile the project
# Run with: quartus_sh -t compile.tcl

# Load required packages
load_package flow

# Check if project exists before opening
if {![project_exists de10_lite_matrix_multiplier]} {
    puts "Error: Project does not exist. Please run create_project.tcl first."
    exit 1
}

# Open project
project_open de10_lite_matrix_multiplier

puts "Starting compilation..."

# Analysis & Synthesis
puts "Running Analysis & Synthesis..."
execute_module -tool map
if {[get_global_assignment -name FLOW_ENABLE_RTL_VIEWER] == "ON"} {
    puts "RTL Viewer enabled - check output"
}

# Fitter (Place & Route)
puts "Running Fitter..."
execute_module -tool fit

# Assembler (Generate programming file)
puts "Running Assembler..."
execute_module -tool asm

# TimeQuest Timing Analyzer
puts "Running Timing Analysis..."
execute_module -tool sta

puts "Compilation complete!"

# Check for timing violations
if {[get_global_assignment -name ENABLE_SIGNALTAP] == "ON"} {
    puts "SignalTap enabled"
}

# Print resource utilization
puts "\n=== Resource Utilization ==="
set total_les [get_fitter_resource -resource "Total logic elements"]
set total_pins [get_fitter_resource -resource "Total pins"]
set total_memory [get_fitter_resource -resource "Total memory bits"]

puts "Logic Elements: $total_les"
puts "Total Pins: $total_pins"  
puts "Memory Bits: $total_memory"

# Close project
project_close

puts "Build process completed successfully!"
puts "Programming file: output_files/de10_lite_matrix_multiplier.sof"