#!/bin/bash
# Script to run FPGA simulation

echo "🧪 Running FPGA Matrix Multiplier Simulation"
echo "============================================="

# Check if we're in the fpga directory
if [ ! -f "de10_lite_top.sv" ]; then
    echo "❌ Error: Must be run from fpga directory"
    exit 1
fi

# Option 1: Try Quartus built-in simulator
if command -v quartus_sh &> /dev/null; then
    echo "📋 Setting up simulation with Quartus..."
    quartus_sh -t simulate.tcl
    
    # Check if simulation directory was created
    if [ -d "simulation" ]; then
        echo "✅ Simulation files generated"
        cd simulation/modelsim 2>/dev/null || cd simulation/questa 2>/dev/null || cd simulation
        
        # Try to run with ModelSim if available
        if command -v vsim &> /dev/null; then
            echo "🚀 Running ModelSim simulation..."
            vsim -c -do "run -all; quit" testbench
        else
            echo "⚠️  ModelSim not found. Simulation files ready in simulation/ directory"
            echo "To run manually:"
            echo "  cd simulation/"
            echo "  vsim testbench"
            echo "  run -all"
        fi
    fi
else
    echo "❌ Quartus not found in PATH"
fi

# Option 2: Try with open-source tools
echo -e "\n🔧 Alternative: Try with Icarus Verilog or Verilator"

# Check for Icarus Verilog
if command -v iverilog &> /dev/null; then
    echo "📋 Found Icarus Verilog, attempting simulation..."
    
    # Compile with iverilog
    iverilog -g2012 -o sim_output \
        testbench.sv \
        de10_lite_top.sv \
        matrix_multiplier.sv \
        uart_controller.sv \
        host_interface.sv \
        pll_100mhz.sv
    
    if [ $? -eq 0 ]; then
        echo "✅ Compilation successful"
        echo "🚀 Running simulation..."
        vvp sim_output
        
        # Check if VCD was generated
        if [ -f "dump.vcd" ]; then
            echo "📊 Waveform saved to dump.vcd"
            echo "💡 View with: gtkwave dump.vcd"
        fi
    else
        echo "❌ Compilation failed"
    fi
    
elif command -v verilator &> /dev/null; then
    echo "📋 Found Verilator, attempting simulation..."
    
    # This would need a C++ testbench for Verilator
    echo "⚠️  Verilator requires C++ testbench (not implemented in this script)"
    echo "💡 Use Icarus Verilog or ModelSim instead"
    
else
    echo "❌ No simulation tools found"
    echo "Please install one of:"
    echo "  • ModelSim/QuestaSim (commercial)"
    echo "  • Icarus Verilog (open source): apt-get install iverilog"
    echo "  • Verilator (open source): apt-get install verilator"
fi

echo -e "\n📋 Simulation Summary:"
echo "✅ Testbench created: testbench.sv"  
echo "✅ Simulation script: simulate.tcl"
echo "💡 Expected result for 2x2 matrix multiplication:"
echo "   [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]"

echo -e "\n🎯 Next steps:"
echo "1. Run simulation with your preferred tool"
echo "2. Verify UART communication works"
echo "3. Check matrix multiplication results"
echo "4. If simulation passes, program actual FPGA board"