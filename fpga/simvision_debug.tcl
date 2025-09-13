# SimVision debugging and analysis script
# Load this after simulation for detailed analysis

# Function to analyze UART traffic
proc analyze_uart {} {
    puts "🔍 Analyzing UART Communication"
    puts "==============================="
    
    # Find UART transactions
    set rx_times [database query -times /testbench/uart_rx -transition 1to0]
    set tx_times [database query -times /testbench/uart_tx -transition 0to1]
    
    puts "UART RX start bits detected: [llength $rx_times]"
    puts "UART TX start bits detected: [llength $tx_times]"
    
    if {[llength $rx_times] > 0} {
        puts "First RX transaction at: [lindex $rx_times 0]"
    }
    
    if {[llength $tx_times] > 0} {
        puts "First TX transaction at: [lindex $tx_times 0]"
    }
}

# Function to check matrix multiplier operation
proc check_matrix_multiplier {} {
    puts "\n🧮 Checking Matrix Multiplier"
    puts "============================="
    
    # Check if MM was started
    set mm_start_times [database query -times /testbench/dut/mm_start -transition 0to1]
    set mm_done_times [database query -times /testbench/dut/mm_done -transition 0to1]
    
    if {[llength $mm_start_times] > 0} {
        puts "✅ Matrix multiplier started at: [lindex $mm_start_times 0]"
    } else {
        puts "❌ Matrix multiplier never started"
    }
    
    if {[llength $mm_done_times] > 0} {
        puts "✅ Matrix multiplier completed at: [lindex $mm_done_times 0]"
        
        # Calculate computation time
        if {[llength $mm_start_times] > 0} {
            set comp_time [expr [lindex $mm_done_times 0] - [lindex $mm_start_times 0]]
            puts "⏱️  Computation time: ${comp_time}ns"
        }
    } else {
        puts "❌ Matrix multiplier never completed"
    }
}

# Function to analyze state machine
proc analyze_state_machine {} {
    puts "\n🔄 Analyzing State Machine"
    puts "=========================="
    
    # Get state transitions
    set states [database query -value /testbench/dut/host_if/current_state -time end]
    puts "Final state: $states"
    
    # Look for state transitions
    set state_changes [database query -times /testbench/dut/host_if/current_state -transition]
    puts "State changes detected: [llength $state_changes]"
    
    foreach change $state_changes {
        set state_val [database query -value /testbench/dut/host_if/current_state -time $change]
        puts "  State change at ${change}ns to: $state_val"
    }
}

# Function to verify expected results
proc verify_results {} {
    puts "\n✅ Verifying Results"
    puts "==================="
    
    # Expected matrix result: [[19,22],[43,50]]
    puts "Expected result matrix C:"
    puts "  [[19, 22],"
    puts "   [43, 50]]"
    
    # Try to read memory contents (if accessible)
    puts "\n💾 Memory Analysis:"
    puts "Check memory interface signals for data written/read"
    
    # Check final LED state
    set final_leds [database query -value /testbench/LEDR -time end]
    puts "Final LED state: $final_leds"
    
    # Check 7-segment displays
    set hex0 [database query -value /testbench/HEX0 -time end]
    set hex1 [database query -value /testbench/HEX1 -time end]
    puts "7-segment displays: HEX0=$hex0, HEX1=$hex1"
}

# Main analysis function
proc run_analysis {} {
    puts "🔬 SimVision Automated Analysis"
    puts "==============================="
    puts "Simulation time: [database query -time end]"
    
    analyze_uart
    check_matrix_multiplier  
    analyze_state_machine
    verify_results
    
    puts "\n🎯 Analysis Complete"
    puts "Check waveforms for detailed timing analysis"
}

# Set up custom waveform views
proc setup_debug_views {} {
    # Create a focused view for UART debugging
    window new WaveWindow -name "UART Debug"
    waveform add -using "UART Debug" /testbench/clk
    waveform add -using "UART Debug" /testbench/uart_rx  
    waveform add -using "UART Debug" /testbench/uart_tx
    waveform add -using "UART Debug" /testbench/dut/uart_ctrl/rx_state
    waveform add -using "UART Debug" /testbench/dut/uart_ctrl/tx_state
    waveform add -using "UART Debug" /testbench/dut/uart_ctrl/uart_rx_data
    waveform add -using "UART Debug" /testbench/dut/uart_ctrl/uart_tx_data
    
    # Create view for matrix multiplier
    window new WaveWindow -name "Matrix Multiplier Debug"
    waveform add -using "Matrix Multiplier Debug" /testbench/clk
    waveform add -using "Matrix Multiplier Debug" /testbench/dut/mm_start
    waveform add -using "Matrix Multiplier Debug" /testbench/dut/mm_done
    waveform add -using "Matrix Multiplier Debug" /testbench/dut/mm_ready
}

# Print available commands
puts "🔧 SimVision Debug Commands Available:"
puts "  run_analysis        - Run automated analysis"
puts "  analyze_uart        - Check UART communication" 
puts "  check_matrix_multiplier - Check MM operation"
puts "  analyze_state_machine   - Check FSM transitions"
puts "  verify_results      - Check final results"
puts "  setup_debug_views   - Create focused debug windows"
puts ""
puts "💡 Usage: Type command name in SimVision console"
puts "Example: run_analysis"