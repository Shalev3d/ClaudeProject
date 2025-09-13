#!/bin/bash
# Create testbench.sv for your existing design

cat > testbench.sv << 'EOF'
`timescale 1ns/1ps

module testbench;

    // Clock and reset
    logic clk;
    logic [1:0] key;
    
    // DE10-Lite board signals
    logic [9:0] sw;
    logic [9:0] ledr; 
    logic [7:0] hex0, hex1, hex2, hex3, hex4, hex5;
    logic [35:0] gpio_0, gpio_1;
    logic [15:0] arduino_io;
    
    // DRAM signals (can be left unconnected)
    logic [12:0] dram_addr;
    logic [1:0] dram_ba;
    logic dram_cas_n, dram_cke, dram_clk, dram_cs_n;
    logic [15:0] dram_dq;
    logic dram_ldqm, dram_ras_n, dram_udqm, dram_we_n;
    
    // Test data
    logic [7:0] test_data;
    integer i, j;
    
    // Instantiate your DE10-Lite top module
    de10_lite_top dut (
        .MAX10_CLK1_50(clk),
        .KEY(key),
        .SW(sw),
        .LEDR(ledr),
        .HEX0(hex0), .HEX1(hex1), .HEX2(hex2),
        .HEX3(hex3), .HEX4(hex4), .HEX5(hex5),
        .GPIO_0(gpio_0),
        .GPIO_1(gpio_1), 
        .ARDUINO_IO(arduino_io),
        .DRAM_ADDR(dram_addr),
        .DRAM_BA(dram_ba),
        .DRAM_CAS_N(dram_cas_n),
        .DRAM_CKE(dram_cke),
        .DRAM_CLK(dram_clk),
        .DRAM_CS_N(dram_cs_n),
        .DRAM_DQ(dram_dq),
        .DRAM_LDQM(dram_ldqm),
        .DRAM_RAS_N(dram_ras_n),
        .DRAM_UDQM(dram_udqm),
        .DRAM_WE_N(dram_we_n)
    );
    
    // Clock generation (50MHz)
    initial begin
        clk = 0;
        forever #10 clk = ~clk;  // 50MHz = 20ns period
    end
    
    // UART transmit task (using ARDUINO_IO pins)
    task uart_send_byte(input [7:0] data);
        begin
            arduino_io[0] = 0;  // Start bit (assuming UART RX is on pin 0)
            #8680;              // Baud rate delay (115200 bps = 8.68us per bit)
            for (i = 0; i < 8; i++) begin
                arduino_io[0] = data[i];
                #8680;
            end
            arduino_io[0] = 1;  // Stop bit
            #8680;
        end
    endtask
    
    // Test sequence
    initial begin
        $display("=== DE10-Lite Matrix Multiplier Testbench ===");
        $display("Starting simulation at time %0t", $time);
        
        // Initialize signals
        key = 2'b10;          // Reset active (KEY[0] = 0)
        sw = 10'h000;         // All switches off
        arduino_io = 16'hFFFF; // UART idle high
        gpio_0 = 36'h0;
        gpio_1 = 36'h0;
        
        // Reset sequence
        #100;
        key = 2'b11;          // Release reset (KEY[0] = 1)
        #100;
        
        $display("Reset completed at time %0t", $time);
        
        // Test basic functionality
        $display("Testing basic functionality...");
        
        // Test switch inputs
        sw = 10'h001;
        #1000;
        sw = 10'h002; 
        #1000;
        sw = 10'h004;
        #1000;
        
        // Test UART communication (if implemented)
        $display("Testing UART communication...");
        
        // Send matrix multiplication command
        uart_send_byte(8'h01);  // CMD_MATMUL
        #1000;
        
        // Send simple dimensions: 2x2 matrices
        uart_send_byte(8'h02);  // rows_a LSB
        uart_send_byte(8'h00);  // rows_a MSB
        uart_send_byte(8'h02);  // cols_a LSB  
        uart_send_byte(8'h00);  // cols_a MSB
        uart_send_byte(8'h02);  // cols_b LSB
        uart_send_byte(8'h00);  // cols_b MSB
        
        $display("Sent dimensions at time %0t", $time);
        
        // Send test matrices
        // Matrix A: [[1, 2], [3, 4]]
        uart_send_byte(8'h01); uart_send_byte(8'h00); // A[0,0] = 1
        uart_send_byte(8'h02); uart_send_byte(8'h00); // A[0,1] = 2
        uart_send_byte(8'h03); uart_send_byte(8'h00); // A[1,0] = 3
        uart_send_byte(8'h04); uart_send_byte(8'h00); // A[1,1] = 4
        
        // Matrix B: [[5, 6], [7, 8]]
        uart_send_byte(8'h05); uart_send_byte(8'h00); // B[0,0] = 5
        uart_send_byte(8'h06); uart_send_byte(8'h00); // B[0,1] = 6
        uart_send_byte(8'h07); uart_send_byte(8'h00); // B[1,0] = 7
        uart_send_byte(8'h08); uart_send_byte(8'h00); // B[1,1] = 8
        
        $display("Sent matrices at time %0t", $time);
        
        // Wait for computation
        #100000;  // Wait 100us for computation
        
        // Monitor final state
        $display("\nFinal state at time %0t:", $time);
        $display("LEDs: 0b%010b (0x%03h)", ledr, ledr);
        $display("Switches: 0b%010b (0x%03h)", sw, sw);
        $display("HEX displays:");
        $display("  HEX5=%02h HEX4=%02h HEX3=%02h", hex5, hex4, hex3);
        $display("  HEX2=%02h HEX1=%02h HEX0=%02h", hex2, hex1, hex0);
        
        // Test status command
        uart_send_byte(8'h03);  // CMD_STATUS
        #10000;
        
        $display("Simulation completed at time %0t", $time);
        $finish;
    end
    
    // Monitor for debugging
    initial begin
        $monitor("Time: %0t | Reset: %b | SW: %03h | LEDs: %03h | HEX0: %02h", 
                 $time, key[0], sw, ledr, hex0);
    end
    
    // Timeout watchdog
    initial begin
        #10000000;  // 10ms timeout
        $display("ERROR: Simulation timeout!");
        $finish;
    end
    
endmodule
EOF

echo "✅ testbench.sv created successfully"