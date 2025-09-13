# SimVision waveform setup script
# This script sets up organized waveform windows

# Create main waveform window
window new WaveWindow -name "Matrix Multiplier Waveforms"

# Add clock and reset signals
waveform add -using WaveWindow /testbench/clk
waveform add -using WaveWindow /testbench/rst_n

# Add UART interface signals
waveform add -using WaveWindow -label "UART Interface" -comment ""
waveform add -using WaveWindow /testbench/uart_rx
waveform add -using WaveWindow /testbench/uart_tx
waveform add -using WaveWindow /testbench/dut/uart_ctrl/uart_rx_valid
waveform add -using WaveWindow /testbench/dut/uart_ctrl/uart_rx_data
waveform add -using WaveWindow /testbench/dut/uart_ctrl/uart_tx_valid
waveform add -using WaveWindow /testbench/dut/uart_ctrl/uart_tx_data

# Add host interface state machine
waveform add -using WaveWindow -label "Host Interface FSM" -comment ""
waveform add -using WaveWindow /testbench/dut/host_if/current_state
waveform add -using WaveWindow /testbench/dut/host_if/current_cmd
waveform add -using WaveWindow /testbench/dut/host_if/matrix_rows_a
waveform add -using WaveWindow /testbench/dut/host_if/matrix_cols_a
waveform add -using WaveWindow /testbench/dut/host_if/matrix_cols_b

# Add matrix multiplier signals
waveform add -using WaveWindow -label "Matrix Multiplier" -comment ""
waveform add -using WaveWindow /testbench/dut/mm_core/start
waveform add -using WaveWindow /testbench/dut/mm_core/done
waveform add -using WaveWindow /testbench/dut/mm_core/ready
waveform add -using WaveWindow /testbench/dut/mm_core/rows_a
waveform add -using WaveWindow /testbench/dut/mm_core/cols_a
waveform add -using WaveWindow /testbench/dut/mm_core/cols_b

# Add memory interface signals
waveform add -using WaveWindow -label "Memory Interface" -comment ""
waveform add -using WaveWindow /testbench/dut/mem_addr_a
waveform add -using WaveWindow /testbench/dut/mem_data_a  
waveform add -using WaveWindow /testbench/dut/mem_write_a
waveform add -using WaveWindow /testbench/dut/mem_addr_b
waveform add -using WaveWindow /testbench/dut/mem_data_b
waveform add -using WaveWindow /testbench/dut/mem_write_b
waveform add -using WaveWindow /testbench/dut/mem_addr_c
waveform add -using WaveWindow /testbench/dut/mem_data_c
waveform add -using WaveWindow /testbench/dut/mem_read_c

# Add LED and display outputs
waveform add -using WaveWindow -label "Board Outputs" -comment ""
waveform add -using WaveWindow /testbench/LEDR
waveform add -using WaveWindow /testbench/HEX0
waveform add -using WaveWindow /testbench/HEX1

# Set up display formats
waveform format /testbench/dut/host_if/current_state -radix symbolic
waveform format /testbench/LEDR -radix binary
waveform format /testbench/dut/uart_ctrl/uart_rx_data -radix hex
waveform format /testbench/dut/uart_ctrl/uart_tx_data -radix hex
waveform format /testbench/dut/mem_data_a -radix unsigned
waveform format /testbench/dut/mem_data_b -radix unsigned  
waveform format /testbench/dut/mem_data_c -radix unsigned

# Set time format
waveform configure -timeunit ns

# Zoom to fit all data
waveform configure -signalnamewidth 200
waveform configure -valuewidth 80

# Add cursors and measurements
cursor new -name "Reset Release" -time 100ns
cursor new -name "Command Start" -time 200ns
cursor new -name "Data Complete" -time 10us

# Save the waveform configuration
waveform save waves_config.tcl

puts "SimVision waveform setup complete"
puts "Use 'run' command to start simulation"
puts "Use 'zoom full' to see entire waveform"