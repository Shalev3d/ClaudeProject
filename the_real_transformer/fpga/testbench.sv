// Testbench for DE10-Lite Matrix Multiplier
// Tests the UART interface and matrix multiplication

`timescale 1ns/1ps

module testbench;

    // Clock and reset
    logic clk;
    logic rst_n;
    
    // UART signals  
    logic uart_tx;
    logic uart_rx;
    
    // DE10-Lite board signals (unused in simulation)
    logic [9:0] SW;
    logic [9:0] LEDR; 
    logic [7:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5;
    
    // Test data
    logic [7:0] test_data;
    integer i, j;
    
    // Instantiate DUT (Device Under Test)
    de10_lite_top dut (
        .MAX10_CLK1_50(clk),
        .KEY({1'b1, rst_n}),  // KEY[0] is reset
        .SW(SW),
        .LEDR(LEDR),
        .HEX0(HEX0), .HEX1(HEX1), .HEX2(HEX2),
        .HEX3(HEX3), .HEX4(HEX4), .HEX5(HEX5),
        .GPIO_0(), .GPIO_1(),  // Leave GPIO unconnected
        .ARDUINO_IO(),
        // Connect UART
        .ARDUINO_IO[0](uart_rx),  // RX
        .ARDUINO_IO[1](uart_tx),  // TX
        // DRAM signals (leave unconnected for simulation)
        .DRAM_ADDR(), .DRAM_BA(), .DRAM_CAS_N(), .DRAM_CKE(), .DRAM_CLK(),
        .DRAM_CS_N(), .DRAM_DQ(), .DRAM_LDQM(), .DRAM_RAS_N(), .DRAM_UDQM(),
        .DRAM_WE_N()
    );
    
    // Clock generation (50MHz)
    initial begin
        clk = 0;
        forever #10 clk = ~clk;  // 50MHz = 20ns period
    end
    
    // UART transmit task
    task uart_send_byte(input [7:0] data);
        begin
            uart_rx = 0;  // Start bit
            #8680;        // Baud rate delay (115200 bps)
            for (i = 0; i < 8; i++) begin
                uart_rx = data[i];
                #8680;
            end
            uart_rx = 1;  // Stop bit
            #8680;
        end
    endtask
    
    // UART receive task
    task uart_receive_byte(output [7:0] data);
        begin
            @(negedge uart_tx);  // Wait for start bit
            #4340;               // Wait half bit time
            #8680;               // Skip start bit
            for (i = 0; i < 8; i++) begin
                data[i] = uart_tx;
                #8680;
            end
            // Stop bit check could go here
        end
    endtask
    
    // Test sequence
    initial begin
        $display("=== DE10-Lite Matrix Multiplier Testbench ===");
        $display("Starting simulation at time %0t", $time);
        
        // Initialize signals
        rst_n = 0;
        uart_rx = 1;  // UART idle high
        SW = 10'h000;
        
        // Reset sequence
        #100;
        rst_n = 1;
        #100;
        
        $display("Reset completed at time %0t", $time);
        
        // Test 1: Send matrix multiplication command
        $display("Test 1: Matrix Multiplication Command");
        uart_send_byte(8'h01);  // CMD_MATMUL
        
        // Send dimensions: 2x2 * 2x2 matrices
        uart_send_byte(8'h02);  // rows_a (LSB)
        uart_send_byte(8'h00);  // rows_a (MSB) 
        uart_send_byte(8'h02);  // cols_a (LSB)
        uart_send_byte(8'h00);  // cols_a (MSB)
        uart_send_byte(8'h02);  // cols_b (LSB)
        uart_send_byte(8'h00);  // cols_b (MSB)
        
        $display("Sent dimensions: 2x2 * 2x2 at time %0t", $time);
        
        // Send Matrix A: [[1, 2], [3, 4]]
        uart_send_byte(8'h01);  // A[0,0] = 1 (LSB)
        uart_send_byte(8'h00);  // A[0,0] = 1 (MSB)
        uart_send_byte(8'h02);  // A[0,1] = 2 (LSB)
        uart_send_byte(8'h00);  // A[0,1] = 2 (MSB)
        uart_send_byte(8'h03);  // A[1,0] = 3 (LSB)
        uart_send_byte(8'h00);  // A[1,0] = 3 (MSB)
        uart_send_byte(8'h04);  // A[1,1] = 4 (LSB)
        uart_send_byte(8'h00);  // A[1,1] = 4 (MSB)
        
        $display("Sent Matrix A at time %0t", $time);
        
        // Send Matrix B: [[5, 6], [7, 8]]
        uart_send_byte(8'h05);  // B[0,0] = 5 (LSB)
        uart_send_byte(8'h00);  // B[0,0] = 5 (MSB)
        uart_send_byte(8'h06);  // B[0,1] = 6 (LSB)
        uart_send_byte(8'h00);  // B[0,1] = 6 (MSB)
        uart_send_byte(8'h07);  // B[1,0] = 7 (LSB)
        uart_send_byte(8'h00);  // B[1,0] = 7 (MSB)
        uart_send_byte(8'h08);  // B[1,1] = 8 (LSB)
        uart_send_byte(8'h00);  // B[1,1] = 8 (MSB)
        
        $display("Sent Matrix B at time %0t", $time);
        
        // Wait for computation
        $display("Waiting for matrix multiplication...");
        #100000;  // Wait 100us for computation
        
        // Try to receive result
        $display("Attempting to receive result...");
        repeat(8) begin  // Try to receive 8 bytes (4 16-bit results)
            uart_receive_byte(test_data);
            $display("Received: 0x%02h at time %0t", test_data, $time);
        end
        
        // Test 2: Status command
        $display("\nTest 2: Status Command");
        uart_send_byte(8'h03);  // CMD_STATUS
        #1000;
        
        // Try to receive status
        uart_receive_byte(test_data);
        $display("Status response: 0x%02h", test_data);
        
        // Monitor LEDs and 7-segment displays
        $display("\nFinal state:");
        $display("LEDs: 0b%10b", LEDR);
        $display("HEX displays: %02h %02h %02h %02h %02h %02h", 
                 HEX5, HEX4, HEX3, HEX2, HEX1, HEX0);
        
        #10000;
        $display("Simulation completed at time %0t", $time);
        $finish;
    end
    
    // Monitor for debugging
    initial begin
        $monitor("Time: %0t | Reset: %b | UART_RX: %b | UART_TX: %b | LEDs: %04b", 
                 $time, rst_n, uart_rx, uart_tx, LEDR[3:0]);
    end
    
    // Timeout watchdog
    initial begin
        #1000000;  // 1ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end
    
endmodule