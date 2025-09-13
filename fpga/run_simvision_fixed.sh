#!/bin/bash
# Fixed script to run FPGA simulation with SimVision

echo "🔬 Running FPGA Matrix Multiplier Simulation with SimVision (Fixed)"
echo "================================================================="

# Check if we're in the fpga directory
if [ ! -f "de10_lite_top.sv" ]; then
    echo "❌ Error: Must be run from fpga directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: de10_lite_top.sv, testbench.sv"
    exit 1
fi

# Check if SimVision/xmverilog tools are available
if ! command -v xmvlog &> /dev/null; then
    echo "❌ Error: Cadence xmvlog not found in PATH"
    echo "Make sure Cadence tools are sourced"
    exit 1
fi

echo "✅ Found Cadence tools"

# Create simulation directory
mkdir -p sim_output
cd sim_output

echo -e "\n🧪 Starting Simulation Process"
echo "=============================="

# Step 1: Create proper library setup
echo "1️⃣  Setting up simulation libraries..."

# Create hdl.var file
cat > hdl.var << 'EOF'
# HDL.VAR file for simulation
DEFINE work work
DEFINE worklib work
EOF

# Create cds.lib file  
cat > cds.lib << 'EOF'
# CDS.LIB file for simulation
DEFINE work work
EOF

# Create work library
mkdir -p work

echo "✅ Library setup complete"

# Step 2: Compile SystemVerilog files
echo -e "\n2️⃣  Compiling SystemVerilog files..."
xmvlog -sv \
    -work work \
    -cdslib ./cds.lib \
    -hdlvar ./hdl.var \
    ../testbench.sv \
    ../de10_lite_top.sv \
    ../matrix_multiplier.sv \
    ../uart_controller.sv \
    ../host_interface.sv \
    ../pll_100mhz.sv

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    echo "Checking for missing files..."
    ls -la ../testbench.sv ../de10_lite_top.sv ../matrix_multiplier.sv ../uart_controller.sv ../host_interface.sv ../pll_100mhz.sv
    exit 1
fi
echo "✅ Compilation successful"

# Step 3: Elaborate the design
echo -e "\n3️⃣  Elaborating design..."
xmelab -work work -cdslib ./cds.lib -hdlvar ./hdl.var testbench:module

if [ $? -ne 0 ]; then
    echo "❌ Elaboration failed!"
    echo "Trying alternative elaboration..."
    xmelab work.testbench:sv
    if [ $? -ne 0 ]; then
        echo "❌ Alternative elaboration also failed!"
        exit 1
    fi
fi
echo "✅ Elaboration successful"

# Step 4: Create simulation script
echo -e "\n4️⃣  Creating simulation commands..."
cat > sim_commands.tcl << 'EOF'
# SimVision simulation commands

# Set up database for waveform viewing  
database -open waves -into waves.shm -default

# Probe all signals for waveform viewing
probe -create -shm testbench -all -variables -depth all

# Start SimVision (if available)
if {[info exists env(DISPLAY)]} {
    simvision waves.shm &
    after 2000
}

# Run the simulation
puts "🚀 Starting simulation run..."
run

# Print results
puts "📊 Simulation Results:"
puts "=============================="
if {[info exists testbench.LEDR]} {
    puts "Final LED state: [examine -value testbench.LEDR]"
}
if {[info exists testbench.HEX0]} {
    puts "HEX0 display: [examine -value testbench.HEX0]"  
}
if {[info exists testbench.uart_tx]} {
    puts "UART TX final: [examine -value testbench.uart_tx]"
}

puts "✅ Simulation completed successfully"
puts "💡 Check SimVision waveform window for detailed analysis"

# Save waveform database
database -save waves
EOF

# Step 5: Run simulation with SimVision
echo -e "\n5️⃣  Starting simulation..."

# Try xmsim first (newer tool)
if command -v xmsim &> /dev/null; then
    echo "Using xmsim..."
    xmsim -gui work.testbench:sv -input sim_commands.tcl
elif command -v imc &> /dev/null; then
    echo "Using imc..."  
    imc -gui -64 work.testbench:sv -input sim_commands.tcl
else
    echo "Running without GUI..."
    # Fallback to command line
    cat > run_sim.tcl << 'EOF'
database -open waves -into waves.shm -default
probe -create -shm testbench -all -variables -depth all  
run
puts "Simulation completed"
exit
EOF
    
    if command -v xmsim &> /dev/null; then
        xmsim work.testbench:sv -input run_sim.tcl
    else
        echo "❌ No suitable simulator found"
        exit 1
    fi
fi

echo -e "\n📋 Simulation Summary"
echo "==================="
echo "✅ Files compiled and elaborated successfully"
echo "✅ Working library created: work/"
echo "✅ Simulation database: waves.shm"

cd ..
echo "🎉 Simulation setup complete!"