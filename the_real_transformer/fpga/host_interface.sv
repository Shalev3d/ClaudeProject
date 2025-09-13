// Host Interface Module for Matrix Multiplier
// Handles communication protocol with Python host

module host_interface (
    input  logic        clk,
    input  logic        rst_n,
    
    // UART interface
    input  logic [7:0]  uart_rx_data,
    input  logic        uart_rx_valid,
    output logic        uart_rx_ready,
    
    output logic [7:0]  uart_tx_data,
    output logic        uart_tx_valid,
    input  logic        uart_tx_ready,
    
    // Matrix multiplier interface
    output logic        mm_start,
    output logic [15:0] mm_rows_a,
    output logic [15:0] mm_cols_a,
    output logic [15:0] mm_cols_b,
    input  logic        mm_done,
    input  logic        mm_ready,
    
    // Memory interface for matrix data
    output logic [9:0]  mem_addr_a,
    output logic [15:0] mem_data_a,
    output logic        mem_write_a,
    
    output logic [9:0]  mem_addr_b,
    output logic [15:0] mem_data_b,
    output logic        mem_write_b,
    
    output logic [9:0]  mem_addr_c,
    input  logic [15:0] mem_data_c,
    output logic        mem_read_c,
    
    output logic [31:0] operation_count
);

// Command definitions
localparam [7:0] CMD_MATMUL = 8'h01;
localparam [7:0] CMD_RESET  = 8'h02;
localparam [7:0] CMD_STATUS = 8'h03;

// Response codes
localparam [7:0] RESP_READY    = 8'h01;
localparam [7:0] RESP_BUSY     = 8'h02;
localparam [7:0] RESP_DONE     = 8'hFF;
localparam [7:0] RESP_ERROR    = 8'hEE;

// State machine
typedef enum logic [7:0] {
    IDLE           = 8'h00,
    RECV_CMD       = 8'h01,
    RECV_DIMS      = 8'h02,
    RECV_MATRIX_A  = 8'h03,
    RECV_MATRIX_B  = 8'h04,
    START_COMPUTE  = 8'h05,
    WAIT_COMPUTE   = 8'h06,
    SEND_RESULT    = 8'h07,
    SEND_RESPONSE  = 8'h08,
    ERROR_STATE    = 8'hEE
} state_t;

state_t current_state, next_state;

// Protocol variables
logic [7:0]  current_cmd;
logic [15:0] matrix_rows_a;
logic [15:0] matrix_cols_a; 
logic [15:0] matrix_cols_b;
logic [15:0] element_counter;
logic [15:0] total_elements;
logic [9:0]  current_addr;

// Data reception buffer
logic [7:0]  rx_buffer [4:0];  // Increased size to avoid index 4 out of bounds
logic [2:0]  rx_buffer_count;

// Data transmission buffer  
logic [7:0]  tx_buffer [3:0];
logic [2:0]  tx_buffer_count;
logic [2:0]  tx_buffer_index;

// Control signals
logic load_matrix_a;
logic load_matrix_b;
logic read_result_c;
logic send_completion;

// FSM
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_state <= IDLE;
    end else begin
        current_state <= next_state;
    end
end

always_comb begin
    next_state = current_state;
    
    case (current_state)
        IDLE: begin
            if (uart_rx_valid) begin
                next_state = RECV_CMD;
            end
        end
        
        RECV_CMD: begin
            if (uart_rx_valid) begin
                case (uart_rx_data)
                    CMD_MATMUL: next_state = RECV_DIMS;
                    CMD_RESET:  next_state = IDLE;
                    CMD_STATUS: next_state = SEND_RESPONSE;
                    default:    next_state = ERROR_STATE;
                endcase
            end
        end
        
        RECV_DIMS: begin
            if (rx_buffer_count == 5) begin  // Received all dimension bytes
                next_state = RECV_MATRIX_A;
            end
        end
        
        RECV_MATRIX_A: begin
            if (element_counter == matrix_rows_a * matrix_cols_a - 1) begin
                next_state = RECV_MATRIX_B;
            end
        end
        
        RECV_MATRIX_B: begin
            if (element_counter == matrix_cols_a * matrix_cols_b - 1) begin
                next_state = START_COMPUTE;
            end
        end
        
        START_COMPUTE: begin
            if (mm_ready) begin
                next_state = WAIT_COMPUTE;
            end
        end
        
        WAIT_COMPUTE: begin
            if (mm_done) begin
                next_state = SEND_RESULT;
            end
        end
        
        SEND_RESULT: begin
            if (element_counter == matrix_rows_a * matrix_cols_b - 1 && 
                tx_buffer_count == 0) begin
                next_state = SEND_RESPONSE;
            end
        end
        
        SEND_RESPONSE: begin
            if (uart_tx_ready && !uart_tx_valid) begin
                next_state = IDLE;
            end
        end
        
        ERROR_STATE: begin
            if (uart_tx_ready && !uart_tx_valid) begin
                next_state = IDLE;
            end
        end
    endcase
end

// State machine control logic
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        current_cmd <= 0;
        matrix_rows_a <= 0;
        matrix_cols_a <= 0;
        matrix_cols_b <= 0;
        element_counter <= 0;
        current_addr <= 0;
        rx_buffer_count <= 0;
        tx_buffer_count <= 0;
        tx_buffer_index <= 0;
        operation_count <= 0;
        
        // Control signals
        mm_start <= 0;
        mm_rows_a <= 0;
        mm_cols_a <= 0;
        mm_cols_b <= 0;
        
        // Memory interface
        mem_addr_a <= 0;
        mem_data_a <= 0;
        mem_write_a <= 0;
        mem_addr_b <= 0;
        mem_data_b <= 0;
        mem_write_b <= 0;
        mem_addr_c <= 0;
        mem_read_c <= 0;
        
        // UART
        uart_tx_data <= 0;
        uart_tx_valid <= 0;
        uart_rx_ready <= 1;
        
    end else begin
        // Default values to prevent latch inference
        mm_start <= 0;
        mem_write_a <= 0;
        mem_write_b <= 0;
        mem_read_c <= 0;
        uart_tx_valid <= 0;
        
        // Default assignments for all outputs (prevent latch inference)
        operation_count <= operation_count;
        mm_rows_a <= mm_rows_a;
        mm_cols_a <= mm_cols_a;
        mm_cols_b <= mm_cols_b;
        mem_addr_a <= mem_addr_a;
        mem_data_a <= mem_data_a;
        mem_addr_b <= mem_addr_b;
        mem_data_b <= mem_data_b;
        mem_addr_c <= mem_addr_c;
        uart_tx_data <= uart_tx_data;
        
        case (current_state)
            IDLE: begin
                uart_rx_ready <= 1;
                element_counter <= 0;
                current_addr <= 0;
                rx_buffer_count <= 0;
                tx_buffer_count <= 0;
                tx_buffer_index <= 0;
            end
            
            RECV_CMD: begin
                if (uart_rx_valid && uart_rx_ready) begin
                    current_cmd <= uart_rx_data;
                    rx_buffer_count <= 0;
                end
            end
            
            RECV_DIMS: begin
                if (uart_rx_valid && uart_rx_ready) begin
                    rx_buffer[rx_buffer_count] <= uart_rx_data;
                    rx_buffer_count <= rx_buffer_count + 1;
                    
                    // Parse dimensions after receiving all bytes
                    if (rx_buffer_count == 5) begin
                        matrix_rows_a <= {rx_buffer[1], rx_buffer[0]};
                        matrix_cols_a <= {rx_buffer[3], rx_buffer[2]};
                        matrix_cols_b <= {uart_rx_data, rx_buffer[4]};
                        element_counter <= 0;
                        current_addr <= 0;
                    end
                end
            end
            
            RECV_MATRIX_A: begin
                if (uart_rx_valid && uart_rx_ready) begin
                    rx_buffer[rx_buffer_count] <= uart_rx_data;
                    rx_buffer_count <= rx_buffer_count + 1;
                    
                    // Write 16-bit element to memory
                    if (rx_buffer_count == 1) begin
                        mem_addr_a <= current_addr;
                        mem_data_a <= {uart_rx_data, rx_buffer[0]};
                        mem_write_a <= 1;
                        
                        current_addr <= current_addr + 1;
                        element_counter <= element_counter + 1;
                        rx_buffer_count <= 0;
                    end
                end
            end
            
            RECV_MATRIX_B: begin
                if (uart_rx_valid && uart_rx_ready) begin
                    rx_buffer[rx_buffer_count] <= uart_rx_data;
                    rx_buffer_count <= rx_buffer_count + 1;
                    
                    // Write 16-bit element to memory
                    if (rx_buffer_count == 1) begin
                        mem_addr_b <= current_addr;
                        mem_data_b <= {uart_rx_data, rx_buffer[0]};
                        mem_write_b <= 1;
                        
                        current_addr <= current_addr + 1;
                        element_counter <= element_counter + 1;
                        rx_buffer_count <= 0;
                    end
                end
            end
            
            START_COMPUTE: begin
                if (mm_ready) begin
                    mm_rows_a <= matrix_rows_a;
                    mm_cols_a <= matrix_cols_a;
                    mm_cols_b <= matrix_cols_b;
                    mm_start <= 1;
                    operation_count <= operation_count + 1;
                end
            end
            
            WAIT_COMPUTE: begin
                // Wait for matrix multiplication to complete
            end
            
            SEND_RESULT: begin
                // Read result matrix and send via UART
                if (tx_buffer_count == 0 && uart_tx_ready) begin
                    mem_addr_c <= element_counter;
                    mem_read_c <= 1;
                end else if (tx_buffer_count > 0 && uart_tx_ready) begin
                    uart_tx_data <= tx_buffer[tx_buffer_index];
                    uart_tx_valid <= 1;
                    tx_buffer_index <= tx_buffer_index + 1;
                    
                    if (tx_buffer_index == tx_buffer_count - 1) begin
                        tx_buffer_count <= 0;
                        tx_buffer_index <= 0;
                        element_counter <= element_counter + 1;
                    end
                end
                
                // Buffer result data
                if (mem_read_c) begin
                    tx_buffer[0] <= mem_data_c[7:0];
                    tx_buffer[1] <= mem_data_c[15:8];
                    tx_buffer_count <= 2;
                    tx_buffer_index <= 0;
                end
            end
            
            SEND_RESPONSE: begin
                if (uart_tx_ready && !uart_tx_valid) begin
                    case (current_cmd)
                        CMD_STATUS: uart_tx_data <= mm_ready ? RESP_READY : RESP_BUSY;
                        CMD_MATMUL: uart_tx_data <= RESP_DONE;
                        default:    uart_tx_data <= RESP_ERROR;
                    endcase
                    uart_tx_valid <= 1;
                end
            end
            
            ERROR_STATE: begin
                if (uart_tx_ready && !uart_tx_valid) begin
                    uart_tx_data <= RESP_ERROR;
                    uart_tx_valid <= 1;
                end
            end
        endcase
    end
end

endmodule