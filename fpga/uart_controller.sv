// UART Controller for DE10-Lite Matrix Multiplier
// Handles serial communication with host PC

module uart_controller #(
    parameter CLOCK_FREQ = 50000000,  // 50 MHz
    parameter BAUD_RATE = 115200
)(
    input  logic       clk,
    input  logic       rst_n,
    
    // UART physical interface
    input  logic       rx,
    output logic       tx,
    
    // Internal interface
    output logic [7:0] rx_data,
    output logic       rx_valid,
    input  logic       rx_ready,
    
    input  logic [7:0] tx_data,
    input  logic       tx_valid,
    output logic       tx_ready
);

localparam int BAUD_TICK = CLOCK_FREQ / BAUD_RATE;
localparam int BAUD_COUNTER_WIDTH = $clog2(BAUD_TICK);

// UART Transmitter
logic [BAUD_COUNTER_WIDTH-1:0] tx_baud_counter;
logic tx_baud_tick;
logic [3:0] tx_bit_counter;
logic [9:0] tx_shift_reg;  // Start bit + 8 data bits + stop bit
logic tx_active;

// UART Receiver  
logic [BAUD_COUNTER_WIDTH-1:0] rx_baud_counter;
logic rx_baud_tick;
logic [3:0] rx_bit_counter;
logic [7:0] rx_shift_reg;
logic rx_active;
logic rx_sync;
logic rx_d1, rx_d2;  // Synchronizer

// Baud rate generation for transmitter
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tx_baud_counter <= 0;
        tx_baud_tick <= 0;
    end else begin
        if (tx_baud_counter == BAUD_TICK - 1) begin
            tx_baud_counter <= 0;
            tx_baud_tick <= 1;
        end else begin
            tx_baud_counter <= tx_baud_counter + 1;
            tx_baud_tick <= 0;
        end
    end
end

// UART Transmitter FSM
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        tx <= 1;
        tx_ready <= 1;
        tx_active <= 0;
        tx_bit_counter <= 0;
        tx_shift_reg <= 10'h3FF;  // All ones (idle)
    end else begin
        if (!tx_active && tx_valid && tx_ready) begin
            // Start transmission
            tx_shift_reg <= {1'b1, tx_data, 1'b0};  // Stop bit + data + start bit
            tx_active <= 1;
            tx_ready <= 0;
            tx_bit_counter <= 0;
        end else if (tx_active && tx_baud_tick) begin
            // Shift out bits
            tx <= tx_shift_reg[0];
            tx_shift_reg <= {1'b1, tx_shift_reg[9:1]};
            
            if (tx_bit_counter == 9) begin
                // Transmission complete
                tx_active <= 0;
                tx_ready <= 1;
                tx_bit_counter <= 0;
            end else begin
                tx_bit_counter <= tx_bit_counter + 1;
            end
        end
    end
end

// RX synchronizer
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rx_d1 <= 1;
        rx_d2 <= 1;
        rx_sync <= 1;
    end else begin
        rx_d1 <= rx;
        rx_d2 <= rx_d1;
        rx_sync <= rx_d2;
    end
end

// Baud rate generation for receiver (16x oversampling)
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rx_baud_counter <= 0;
        rx_baud_tick <= 0;
    end else begin
        if (rx_baud_counter == (BAUD_TICK / 16) - 1) begin
            rx_baud_counter <= 0;
            rx_baud_tick <= 1;
        end else begin
            rx_baud_counter <= rx_baud_counter + 1;
            rx_baud_tick <= 0;
        end
    end
end

// UART Receiver FSM
logic [3:0] rx_sample_counter;
logic rx_start_detected;

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        rx_active <= 0;
        rx_bit_counter <= 0;
        rx_sample_counter <= 0;
        rx_shift_reg <= 0;
        rx_data <= 0;
        rx_valid <= 0;
        rx_start_detected <= 0;
    end else begin
        rx_valid <= 0;  // Default
        
        if (rx_baud_tick) begin
            if (!rx_active && !rx_sync) begin
                // Start bit detected
                rx_active <= 1;
                rx_bit_counter <= 0;
                rx_sample_counter <= 0;
                rx_start_detected <= 1;
            end else if (rx_active) begin
                rx_sample_counter <= rx_sample_counter + 1;
                
                // Sample at middle of bit period (8th tick out of 16)
                if (rx_sample_counter == 7) begin
                    if (rx_start_detected) begin
                        // Verify start bit
                        if (!rx_sync) begin
                            rx_start_detected <= 0;
                            rx_bit_counter <= 0;
                        end else begin
                            // False start, reset
                            rx_active <= 0;
                            rx_start_detected <= 0;
                        end
                    end else if (rx_bit_counter < 8) begin
                        // Sample data bits
                        rx_shift_reg <= {rx_sync, rx_shift_reg[7:1]};
                        rx_bit_counter <= rx_bit_counter + 1;
                    end else begin
                        // Stop bit - frame complete
                        if (rx_sync) begin  // Valid stop bit
                            rx_data <= rx_shift_reg;
                            rx_valid <= 1;
                        end
                        rx_active <= 0;
                        rx_bit_counter <= 0;
                    end
                end
                
                if (rx_sample_counter == 15) begin
                    rx_sample_counter <= 0;
                end
            end
        end
        
        // Clear rx_valid when acknowledged
        if (rx_valid && rx_ready) begin
            rx_valid <= 0;
        end
    end
end

endmodule