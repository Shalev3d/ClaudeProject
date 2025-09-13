# Synopsys Design Constraints (SDC) for DE10-Lite Matrix Multiplier
# DE10-Lite board constraints for transformer accelerator

# Create clock constraints
create_clock -name "clk_50" -period 20.000ns [get_ports {MAX10_CLK1_50}]
create_clock -name "clk_50_2" -period 20.000ns [get_ports {MAX10_CLK2_50}]

# Derive PLL clocks (if using PLLs for faster operation)
derive_pll_clocks
derive_clock_uncertainty

# Set input/output delays for external interfaces
set_input_delay -clock [get_clocks {clk_50}] -max 5.000ns [get_ports {SW[*]}]
set_input_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {SW[*]}]

set_input_delay -clock [get_clocks {clk_50}] -max 5.000ns [get_ports {KEY[*]}]
set_input_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {KEY[*]}]

set_output_delay -clock [get_clocks {clk_50}] -max 5.000ns [get_ports {LEDR[*]}]
set_output_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {LEDR[*]}]

set_output_delay -clock [get_clocks {clk_50}] -max 5.000ns [get_ports {HEX*[*]}]
set_output_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {HEX*[*]}]

# UART interface timing constraints
set_input_delay -clock [get_clocks {clk_50}] -max 2.000ns [get_ports {UART_RXD}]
set_input_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {UART_RXD}]

set_output_delay -clock [get_clocks {clk_50}] -max 2.000ns [get_ports {UART_TXD}]
set_output_delay -clock [get_clocks {clk_50}] -min 0.000ns [get_ports {UART_TXD}]

# Set false paths for asynchronous signals
set_false_path -from [get_ports {KEY[0]}] -to *
set_false_path -from [get_ports {SW[*]}] -to *

# SDRAM interface constraints (if using external memory)
if {[llength [get_ports -quiet {DRAM_*}]] > 0} {
    # SDRAM clock
    create_generated_clock -name "sdram_clk" -source [get_pins {*|pll_inst|altera_pll_i|*[0].*|divclk}] \
                          -multiply_by 1 [get_ports {DRAM_CLK}]
    
    # SDRAM input timing
    set_input_delay -clock [get_clocks {sdram_clk}] -max 6.4ns [get_ports {DRAM_DQ[*]}]
    set_input_delay -clock [get_clocks {sdram_clk}] -min 3.2ns [get_ports {DRAM_DQ[*]}]
    
    # SDRAM output timing
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_DQ[*]}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_DQ[*]}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_ADDR[*]}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_ADDR[*]}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_BA[*]}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_BA[*]}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_*WE_N}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_*WE_N}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_*CAS_N}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_*CAS_N}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_*RAS_N}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_*RAS_N}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_CS_N}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_CS_N}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_CKE}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_CKE}]
    
    set_output_delay -clock [get_clocks {sdram_clk}] -max 1.5ns [get_ports {DRAM_DQM[*]}]
    set_output_delay -clock [get_clocks {sdram_clk}] -min -0.8ns [get_ports {DRAM_DQM[*]}]
}

# Set multicycle paths for slower operations
set_multicycle_path -from [get_registers {*matrix_multiplier*}] -to [get_registers {*matrix_multiplier*}] -setup 2
set_multicycle_path -from [get_registers {*matrix_multiplier*}] -to [get_registers {*matrix_multiplier*}] -hold 1

# Set maximum fanout to improve timing
set_max_fanout 20 [current_design]

# Set maximum transition time
set_max_transition 2.0ns [current_design]

# Cut combinational loops if any
set_logic_dc [get_pins -hierarchical *]

# Power optimization
set_power_file_mode on