// Top-level module for DE10-Lite FPGA Matrix Multiplier
// Interfaces with host PC via UART and provides matrix multiplication acceleration

module de10_lite_top (
    // Clock inputs
    input  logic        MAX10_CLK1_50,
    input  logic        MAX10_CLK2_50,
    
    // Key inputs (active low)
    input  logic [1:0]  KEY,
    
    // Switch inputs
    input  logic [9:0]  SW,
    
    // LED outputs
    output logic [9:0]  LEDR,
    
    // 7-segment displays
    output logic [7:0]  HEX0,
    output logic [7:0]  HEX1,
    output logic [7:0]  HEX2,
    output logic [7:0]  HEX3,
    output logic [7:0]  HEX4,
    output logic [7:0]  HEX5,
    
    // UART interface
    input  logic        UART_RXD,
    output logic        UART_TXD,
    
    // SDRAM interface (optional for larger matrices)
    output logic        DRAM_CLK,
    output logic        DRAM_CKE,
    output logic [12:0] DRAM_ADDR,
    inout  wire  [15:0] DRAM_DQ,
    output logic        DRAM_LDQM,
    output logic        DRAM_UDQM,
    output logic        DRAM_CS_N,
    output logic        DRAM_WE_N,
    output logic        DRAM_CAS_N,
    output logic        DRAM_RAS_N,
    output logic [1:0]  DRAM_BA
);

// Clock and reset signals
logic clk_50;
logic clk_100;  // Faster clock for matrix operations
logic rst_n;
logic pll_locked;

assign clk_50 = MAX10_CLK1_50;
assign rst_n = KEY[0] & pll_locked;

// PLL for generating faster clock
pll_100mhz pll_inst (
    .inclk0(clk_50),
    .c0(clk_100),
    .locked(pll_locked)
);

// UART controller signals
logic [7:0]  uart_rx_data;
logic        uart_rx_valid;
logic        uart_rx_ready;
logic [7:0]  uart_tx_data;
logic        uart_tx_valid;
logic        uart_tx_ready;

// Matrix multiplier signals
logic        mm_start;
logic [15:0] mm_rows_a;
logic [15:0] mm_cols_a;
logic [15:0] mm_cols_b;
logic        mm_done;
logic        mm_ready;

// Memory interface signals for matrix data
logic [9:0]  mem_addr_a;
logic [15:0] mem_data_a;
logic        mem_read_en_a;

logic [9:0]  mem_addr_b;
logic [15:0] mem_data_b;
logic        mem_read_en_b;

logic [9:0]  mem_addr_c;
logic [15:0] mem_data_c;
logic        mem_write_en_c;

// Debug and status signals
logic [7:0]  debug_state;
logic [15:0] debug_cycles;
logic [31:0] operation_count;

// UART Controller
uart_controller uart_ctrl (
    .clk(clk_50),
    .rst_n(rst_n),
    .rx(UART_RXD),
    .tx(UART_TXD),
    .rx_data(uart_rx_data),
    .rx_valid(uart_rx_valid),
    .rx_ready(uart_rx_ready),
    .tx_data(uart_tx_data),
    .tx_valid(uart_tx_valid),
    .tx_ready(uart_tx_ready)
);

// Matrix Memory Blocks (using internal BRAM)
matrix_memory #(
    .DATA_WIDTH(16),
    .ADDR_WIDTH(10),
    .MEMORY_SIZE(1024)
) mem_a_inst (
    .clk(clk_100),
    .rst_n(rst_n),
    .addr(mem_addr_a),
    .data_in(16'h0000),  // Read-only for matrix A
    .data_out(mem_data_a),
    .write_en(1'b0),
    .read_en(mem_read_en_a)
);

matrix_memory #(
    .DATA_WIDTH(16),
    .ADDR_WIDTH(10),
    .MEMORY_SIZE(1024)
) mem_b_inst (
    .clk(clk_100),
    .rst_n(rst_n),
    .addr(mem_addr_b),
    .data_in(16'h0000),  // Read-only for matrix B
    .data_out(mem_data_b),
    .write_en(1'b0),
    .read_en(mem_read_en_b)
);

matrix_memory #(
    .DATA_WIDTH(16),
    .ADDR_WIDTH(10),
    .MEMORY_SIZE(1024)
) mem_c_inst (
    .clk(clk_100),
    .rst_n(rst_n),
    .addr(mem_addr_c),
    .data_in(mem_data_c),
    .data_out(),  // Write-only for result matrix C
    .write_en(mem_write_en_c),
    .read_en(1'b0)
);

// Matrix Multiplier Core
matrix_multiplier #(
    .DATA_WIDTH(16),
    .MATRIX_SIZE(8),
    .ADDR_WIDTH(10),
    .FIFO_DEPTH(256)
) mm_core (
    .clk(clk_100),
    .rst_n(rst_n),
    .start(mm_start),
    .rows_a(mm_rows_a),
    .cols_a(mm_cols_a),
    .cols_b(mm_cols_b),
    .done(mm_done),
    .ready(mm_ready),
    .addr_a(mem_addr_a),
    .data_a(mem_data_a),
    .read_en_a(mem_read_en_a),
    .addr_b(mem_addr_b),
    .data_b(mem_data_b),
    .read_en_b(mem_read_en_b),
    .addr_c(mem_addr_c),
    .data_c(mem_data_c),
    .write_en_c(mem_write_en_c),
    .debug_state(debug_state),
    .debug_cycles(debug_cycles)
);

// Host Communication Protocol Handler
host_interface host_if (
    .clk(clk_50),
    .rst_n(rst_n),
    
    // UART interface
    .uart_rx_data(uart_rx_data),
    .uart_rx_valid(uart_rx_valid),
    .uart_rx_ready(uart_rx_ready),
    .uart_tx_data(uart_tx_data),
    .uart_tx_valid(uart_tx_valid),
    .uart_tx_ready(uart_tx_ready),
    
    // Matrix multiplier interface
    .mm_start(mm_start),
    .mm_rows_a(mm_rows_a),
    .mm_cols_a(mm_cols_a),
    .mm_cols_b(mm_cols_b),
    .mm_done(mm_done),
    .mm_ready(mm_ready),
    
    // Memory interface for loading/storing matrices
    .mem_addr_a(),    // Connected internally
    .mem_data_a(),    // Connected internally
    .mem_write_a(),   // Connected internally
    .mem_addr_b(),    // Connected internally  
    .mem_data_b(),    // Connected internally
    .mem_write_b(),   // Connected internally
    .mem_addr_c(),    // Connected internally
    .mem_data_c(),    // Connected internally
    .mem_read_c(),    // Connected internally
    
    .operation_count(operation_count)
);

// Status and Debug Display
always_ff @(posedge clk_50 or negedge rst_n) begin
    if (!rst_n) begin
        LEDR <= 10'h000;
    end else begin
        // Display status on LEDs
        LEDR[0] <= mm_ready;
        LEDR[1] <= mm_done;
        LEDR[2] <= uart_rx_valid;
        LEDR[3] <= uart_tx_valid;
        LEDR[4] <= pll_locked;
        LEDR[9:5] <= SW[9:5];  // Echo some switches
    end
end

// 7-segment display controller
seven_segment_display seg_display (
    .clk(clk_50),
    .rst_n(rst_n),
    .value_0(debug_cycles[3:0]),
    .value_1(debug_cycles[7:4]),
    .value_2(debug_cycles[11:8]),
    .value_3(debug_cycles[15:12]),
    .value_4(operation_count[3:0]),
    .value_5(operation_count[7:4]),
    .hex0(HEX0),
    .hex1(HEX1),
    .hex2(HEX2),
    .hex3(HEX3),
    .hex4(HEX4),
    .hex5(HEX5)
);

// SDRAM controller (optional, for larger matrices)
// Uncomment if using external SDRAM
/*
sdram_controller sdram_ctrl (
    .clk_100(clk_100),
    .clk_100_shift(),  // Phase-shifted clock for SDRAM
    .rst_n(rst_n),
    
    // SDRAM interface
    .sdram_clk(DRAM_CLK),
    .sdram_cke(DRAM_CKE),
    .sdram_addr(DRAM_ADDR),
    .sdram_dq(DRAM_DQ),
    .sdram_dqm({DRAM_UDQM, DRAM_LDQM}),
    .sdram_cs_n(DRAM_CS_N),
    .sdram_we_n(DRAM_WE_N),
    .sdram_cas_n(DRAM_CAS_N),
    .sdram_ras_n(DRAM_RAS_N),
    .sdram_ba(DRAM_BA),
    
    // Internal interface
    .addr(),
    .write_data(),
    .read_data(),
    .write_en(),
    .read_en(),
    .ready()
);
*/

endmodule

// Simple memory block using BRAM
module matrix_memory #(
    parameter DATA_WIDTH = 16,
    parameter ADDR_WIDTH = 10,
    parameter MEMORY_SIZE = 1024
)(
    input  logic                     clk,
    input  logic                     rst_n,
    input  logic [ADDR_WIDTH-1:0]    addr,
    input  logic [DATA_WIDTH-1:0]    data_in,
    output logic [DATA_WIDTH-1:0]    data_out,
    input  logic                     write_en,
    input  logic                     read_en
);

logic [DATA_WIDTH-1:0] memory [MEMORY_SIZE-1:0];

always_ff @(posedge clk) begin
    if (write_en) begin
        memory[addr] <= data_in;
    end
    
    if (read_en) begin
        data_out <= memory[addr];
    end
end

endmodule

// 7-segment display decoder
module seven_segment_display (
    input  logic       clk,
    input  logic       rst_n,
    input  logic [3:0] value_0,
    input  logic [3:0] value_1,
    input  logic [3:0] value_2,
    input  logic [3:0] value_3,
    input  logic [3:0] value_4,
    input  logic [3:0] value_5,
    output logic [7:0] hex0,
    output logic [7:0] hex1,
    output logic [7:0] hex2,
    output logic [7:0] hex3,
    output logic [7:0] hex4,
    output logic [7:0] hex5
);

function automatic [7:0] hex_to_seven_seg(input [3:0] hex);
    case (hex)
        4'h0: hex_to_seven_seg = 8'hC0;  // 0
        4'h1: hex_to_seven_seg = 8'hF9;  // 1
        4'h2: hex_to_seven_seg = 8'hA4;  // 2
        4'h3: hex_to_seven_seg = 8'hB0;  // 3
        4'h4: hex_to_seven_seg = 8'h99;  // 4
        4'h5: hex_to_seven_seg = 8'h92;  // 5
        4'h6: hex_to_seven_seg = 8'h82;  // 6
        4'h7: hex_to_seven_seg = 8'hF8;  // 7
        4'h8: hex_to_seven_seg = 8'h80;  // 8
        4'h9: hex_to_seven_seg = 8'h90;  // 9
        4'hA: hex_to_seven_seg = 8'h88;  // A
        4'hB: hex_to_seven_seg = 8'h83;  // b
        4'hC: hex_to_seven_seg = 8'hC6;  // C
        4'hD: hex_to_seven_seg = 8'hA1;  // d
        4'hE: hex_to_seven_seg = 8'h86;  // E
        4'hF: hex_to_seven_seg = 8'h8E;  // F
    endcase
endfunction

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        hex0 <= 8'hFF;
        hex1 <= 8'hFF;
        hex2 <= 8'hFF;
        hex3 <= 8'hFF;
        hex4 <= 8'hFF;
        hex5 <= 8'hFF;
    end else begin
        hex0 <= hex_to_seven_seg(value_0);
        hex1 <= hex_to_seven_seg(value_1);
        hex2 <= hex_to_seven_seg(value_2);
        hex3 <= hex_to_seven_seg(value_3);
        hex4 <= hex_to_seven_seg(value_4);
        hex5 <= hex_to_seven_seg(value_5);
    end
end

endmodule