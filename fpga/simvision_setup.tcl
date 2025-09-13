# SimVision TCL script for FPGA Matrix Multiplier simulation
# Run with: simvision -input simvision_setup.tcl

# Set up the simulation environment
set TOP_MODULE testbench
set DESIGN_FILES {
    testbench.sv
    de10_lite_top.sv
    matrix_multiplier.sv
    uart_controller.sv
    host_interface.sv
    pll_100mhz.sv
}

# Create simulation database
database -open waves -into waves.shm -default

# Compile design files
eval "ncvlog -sv $DESIGN_FILES"

# Elaborate the design  
ncelab -access +rwc $TOP_MODULE

# Run simulation
ncsim $TOP_MODULE -input @"
    # Add all signals to waveform
    probe -create -shm $TOP_MODULE -all -variables -depth all
    
    # Create specific signal groups for easier viewing
    probe -create -shm $TOP_MODULE.dut.uart_ctrl -all -variables -depth 1
    probe -create -shm $TOP_MODULE.dut.mm_core -all -variables -depth 1  
    probe -create -shm $TOP_MODULE.dut.host_if -all -variables -depth 1
    
    # Run the simulation
    run
    
    # Save waveforms
    simvision -input simvision_waves.tcl
"