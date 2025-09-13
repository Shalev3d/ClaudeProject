#!/bin/bash
# Script to run FPGA simulation with SimVision

echo "🔬 Running FPGA Matrix Multiplier Simulation with SimVision"
echo "=========================================================="

# Check if we're in the fpga directory
if [ ! -f "de10_lite_top.sv" ]; then
    echo "❌ Error: Must be run from fpga directory"
    echo "Current directory: $(pwd)"
    echo "Expected files: de10_lite_top.sv, testbench.sv"
    exit 1
fi

# Check if SimVision/ncverilog tools are available
if ! command -v ncvlog &> /dev/null; then
    echo "❌ Error: Cadence ncvlog not found in PATH"
    echo "Make sure Cadence tools are sourced:"
    echo "  source /path/to/cadence/tools/bin/.cshrc"
    echo "  or add to PATH: export PATH=\$PATH:/tools/cadence/..."
    exit 1
fi

echo "✅ Found Cadence tools"
echo "📋 Tool versions:"
ncvlog -version 2>/dev/null | head -3
ncelab -version 2>/dev/null | head -1
ncsim -version 2>/dev/null | head -1

# Create simulation directory
mkdir -p sim_output
cd sim_output

echo -e "\n🧪 Starting Simulation Process"
echo "=============================="

# Step 1: Compile SystemVerilog files
echo "1️⃣  Compiling SystemVerilog files..."
ncvlog -sv \
    ../testbench.sv \
    ../de10_lite_top.sv \
    ../matrix_multiplier.sv \
    ../uart_controller.sv \
    ../host_interface.sv \
    ../pll_100mhz.sv \
    -messages -linedebug

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed!"
    exit 1
fi
echo "✅ Compilation successful"

# Step 2: Elaborate the design
echo -e "\n2️⃣  Elaborating design..."
ncelab testbench -access +rwc -messages

if [ $? -ne 0 ]; then
    echo "❌ Elaboration failed!"
    exit 1
fi
echo "✅ Elaboration successful"

# Step 3: Create simulation script
echo -e "\n3️⃣  Creating simulation commands..."
cat > sim_commands.tcl << 'EOF'
# SimVision simulation commands

# Set up database for waveform viewing
database -open waves -into waves.shm -default

# Probe all signals for waveform viewing
probe -create -shm testbench -all -variables -depth all

# Start SimVision waveform viewer
simvision waves.shm &

# Add signals to waveform (this will be picked up by SimVision)
source ../simvision_waves.tcl

# Run the simulation
echo "🚀 Starting simulation run..."
run

# Print results
echo "📊 Simulation Results:"
echo "=============================="
echo "Final LED state: " $signals(testbench.LEDR)
echo "HEX displays: " $signals(testbench.HEX0) $signals(testbench.HEX1)
echo "UART TX final: " $signals(testbench.uart_tx)
echo "Matrix multiplier done: " $signals(testbench.dut.mm_done)

echo "✅ Simulation completed successfully"
echo "💡 Check SimVision waveform window for detailed analysis"

# Keep simulator open for interactive debugging
echo "🔧 Simulation ready for interactive debugging"
echo "Type 'quit' to exit"
EOF

# Step 4: Run simulation with SimVision
echo -e "\n4️⃣  Starting simulation with SimVision..."
echo "💡 This will open SimVision GUI - check your display"

ncsim testbench -gui -input sim_commands.tcl

echo -e "\n📋 Simulation Summary"
echo "==================="
echo "✅ Files compiled and elaborated successfully"
echo "✅ SimVision waveform database created: waves.shm"
echo "✅ Interactive simulation completed"

echo -e "\n🎯 Expected Results:"
echo "Matrix A = [[1,2],[3,4]], Matrix B = [[5,6],[7,8]]"  
echo "Expected Result C = [[19,22],[43,50]]"
echo ""
echo "📊 Check SimVision waveforms for:"
echo "  • UART communication timing"
echo "  • State machine transitions" 
echo "  • Matrix multiplication correctness"
echo "  • Memory interface operations"

echo -e "\n💡 SimVision Tips:"
echo "  • Use 'zoom full' to see entire simulation"
echo "  • Right-click signals to change display format"
echo "  • Use cursors to measure timing"
echo "  • Save waveform configuration for future use"

# Return to original directory
cd ..

echo -e "\n🎉 Simulation setup complete!"
echo "Check the SimVision window for waveform analysis."