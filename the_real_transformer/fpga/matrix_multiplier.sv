// Matrix Multiplier for DE10-Lite FPGA
// Systolic array implementation optimized for transformer operations
// Supports variable matrix sizes with configurable precision

module matrix_multiplier #(
    parameter DATA_WIDTH = 16,          // 16-bit fixed point
    parameter MATRIX_SIZE = 8,          // 8x8 systolic array
    parameter ADDR_WIDTH = 10,          // Memory address width
    parameter FIFO_DEPTH = 256
)(
    input  logic                    clk,
    input  logic                    rst_n,
    
    // Control interface
    input  logic                    start,
    input  logic [15:0]             rows_a,     // Number of rows in matrix A
    input  logic [15:0]             cols_a,     // Number of columns in matrix A / rows in matrix B
    input  logic [15:0]             cols_b,     // Number of columns in matrix B
    output logic                    done,
    output logic                    ready,
    
    // Memory interface for matrix A
    output logic [ADDR_WIDTH-1:0]   addr_a,
    input  logic [DATA_WIDTH-1:0]   data_a,
    output logic                    read_en_a,
    
    // Memory interface for matrix B
    output logic [ADDR_WIDTH-1:0]   addr_b,
    input  logic [DATA_WIDTH-1:0]   data_b,
    output logic                    read_en_b,
    
    // Memory interface for matrix C (result)
    output logic [ADDR_WIDTH-1:0]   addr_c,
    output logic [DATA_WIDTH-1:0]   data_c,
    output logic                    write_en_c,
    
    // Debug outputs
    output logic [7:0]              debug_state,
    output logic [15:0]             debug_cycles
);

// FSM states
typedef enum logic [7:0] {
    IDLE        = 8'h00,
    INIT        = 8'h01,
    LOAD_A      = 8'h02,
    LOAD_B      = 8'h03,
    COMPUTE     = 8'h04,
    DRAIN       = 8'h05,
    WRITE_C     = 8'h06,
    DONE_STATE  = 8'h07
} state_t;

state_t current_state, next_state;

// Systolic array signals
logic [DATA_WIDTH-1:0]  a_data [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
logic [DATA_WIDTH-1:0]  b_data [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
logic [DATA_WIDTH*2-1:0] c_data [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
logic                   valid_a [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
logic                   valid_b [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];
logic                   valid_c [MATRIX_SIZE-1:0][MATRIX_SIZE-1:0];

// Control counters
logic [15:0] row_counter;
logic [15:0] col_counter;
logic [15:0] k_counter;
logic [15:0] cycle_counter;
logic [15:0] drain_counter;

// Input FIFOs
logic [DATA_WIDTH-1:0] fifo_a_data [MATRIX_SIZE-1:0];
logic [DATA_WIDTH-1:0] fifo_b_data [MATRIX_SIZE-1:0];
logic                  fifo_a_empty [MATRIX_SIZE-1:0];
logic                  fifo_b_empty [MATRIX_SIZE-1:0];
logic                  fifo_a_rd_en [MATRIX_SIZE-1:0];
logic                  fifo_b_rd_en [MATRIX_SIZE-1:0];

// Generate systolic array
genvar i, j;
generate
    for (i = 0; i < MATRIX_SIZE; i++) begin : gen_row
        for (j = 0; j < MATRIX_SIZE; j++) begin : gen_col
            processing_element pe_inst (
                .clk(clk),
                .rst_n(rst_n),
                .a_in(i == 0 ? fifo_a_data[j] : a_data[i-1][j]),
                .b_in(j == 0 ? fifo_b_data[i] : b_data[i][j-1]),
                .c_in(i == 0 && j == 0 ? '0 : 
                      i == 0 ? c_data[i][j-1] :
                      j == 0 ? c_data[i-1][j] : c_data[i-1][j] + c_data[i][j-1] - c_data[i-1][j-1]),
                .valid_a_in(i == 0 ? !fifo_a_empty[j] && fifo_a_rd_en[j] : valid_a[i-1][j]),
                .valid_b_in(j == 0 ? !fifo_b_empty[i] && fifo_b_rd_en[i] : valid_b[i][j-1]),
                .a_out(a_data[i][j]),
                .b_out(b_data[i][j]),
                .c_out(c_data[i][j]),
                .valid_a_out(valid_a[i][j]),
                .valid_b_out(valid_b[i][j]),
                .valid_c_out(valid_c[i][j])
            );
        end
    end
endgenerate

// Generate input FIFOs
generate
    for (i = 0; i < MATRIX_SIZE; i++) begin : gen_fifo
        fifo #(
            .DATA_WIDTH(DATA_WIDTH),
            .DEPTH(FIFO_DEPTH)
        ) fifo_a_inst (
            .clk(clk),
            .rst_n(rst_n),
            .wr_en(read_en_a && (addr_a % MATRIX_SIZE == i) && current_state == LOAD_A),
            .rd_en(fifo_a_rd_en[i]),
            .wr_data(data_a),
            .rd_data(fifo_a_data[i]),
            .empty(fifo_a_empty[i]),
            .full()
        );
        
        fifo #(
            .DATA_WIDTH(DATA_WIDTH),
            .DEPTH(FIFO_DEPTH)
        ) fifo_b_inst (
            .clk(clk),
            .rst_n(rst_n),
            .wr_en(read_en_b && (addr_b % MATRIX_SIZE == i) && current_state == LOAD_B),
            .rd_en(fifo_b_rd_en[i]),
            .wr_data(data_b),
            .rd_data(fifo_b_data[i]),
            .empty(fifo_b_empty[i]),
            .full()
        );
    end
endgenerate

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
        IDLE: if (start) next_state = INIT;
        INIT: next_state = LOAD_A;
        LOAD_A: if (row_counter == rows_a - 1 && k_counter == cols_a - 1) next_state = LOAD_B;
        LOAD_B: if (k_counter == cols_a - 1 && col_counter == cols_b - 1) next_state = COMPUTE;
        COMPUTE: if (cycle_counter >= cols_a + MATRIX_SIZE - 1) next_state = DRAIN;
        DRAIN: if (drain_counter >= MATRIX_SIZE - 1) next_state = WRITE_C;
        WRITE_C: if (row_counter == rows_a - 1 && col_counter == cols_b - 1) next_state = DONE_STATE;
        DONE_STATE: next_state = IDLE;
    endcase
end

// Control logic
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        row_counter <= 0;
        col_counter <= 0;
        k_counter <= 0;
        cycle_counter <= 0;
        drain_counter <= 0;
        addr_a <= 0;
        addr_b <= 0;
        addr_c <= 0;
        read_en_a <= 0;
        read_en_b <= 0;
        write_en_c <= 0;
        data_c <= 0;
        done <= 0;
        ready <= 1;
        for (int i = 0; i < MATRIX_SIZE; i++) begin
            fifo_a_rd_en[i] <= 0;
            fifo_b_rd_en[i] <= 0;
        end
    end else begin
        case (current_state)
            IDLE: begin
                ready <= 1;
                done <= 0;
                row_counter <= 0;
                col_counter <= 0;
                k_counter <= 0;
                cycle_counter <= 0;
                drain_counter <= 0;
            end
            
            INIT: begin
                ready <= 0;
                addr_a <= 0;
                addr_b <= 0;
                addr_c <= 0;
            end
            
            LOAD_A: begin
                read_en_a <= 1;
                addr_a <= row_counter * cols_a + k_counter;
                
                if (k_counter == cols_a - 1) begin
                    k_counter <= 0;
                    if (row_counter == rows_a - 1) begin
                        row_counter <= 0;
                        read_en_a <= 0;
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end else begin
                    k_counter <= k_counter + 1;
                end
            end
            
            LOAD_B: begin
                read_en_b <= 1;
                addr_b <= k_counter * cols_b + col_counter;
                
                if (col_counter == cols_b - 1) begin
                    col_counter <= 0;
                    if (k_counter == cols_a - 1) begin
                        k_counter <= 0;
                        read_en_b <= 0;
                    end else begin
                        k_counter <= k_counter + 1;
                    end
                end else begin
                    col_counter <= col_counter + 1;
                end
            end
            
            COMPUTE: begin
                cycle_counter <= cycle_counter + 1;
                
                // Enable FIFO reads based on timing
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    fifo_a_rd_en[i] <= !fifo_a_empty[i] && cycle_counter >= i;
                    fifo_b_rd_en[i] <= !fifo_b_empty[i] && cycle_counter >= i;
                end
            end
            
            DRAIN: begin
                drain_counter <= drain_counter + 1;
                for (int i = 0; i < MATRIX_SIZE; i++) begin
                    fifo_a_rd_en[i] <= 0;
                    fifo_b_rd_en[i] <= 0;
                end
            end
            
            WRITE_C: begin
                write_en_c <= 1;
                addr_c <= row_counter * cols_b + col_counter;
                data_c <= c_data[row_counter % MATRIX_SIZE][col_counter % MATRIX_SIZE][DATA_WIDTH-1:0];
                
                if (col_counter == cols_b - 1) begin
                    col_counter <= 0;
                    if (row_counter == rows_a - 1) begin
                        row_counter <= 0;
                        write_en_c <= 0;
                    end else begin
                        row_counter <= row_counter + 1;
                    end
                end else begin
                    col_counter <= col_counter + 1;
                end
            end
            
            DONE_STATE: begin
                done <= 1;
                write_en_c <= 0;
            end
        endcase
    end
end

// Debug outputs
assign debug_state = current_state;
assign debug_cycles = cycle_counter;

endmodule

// Processing Element for systolic array
module processing_element #(
    parameter DATA_WIDTH = 16
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic [DATA_WIDTH-1:0]   a_in,
    input  logic [DATA_WIDTH-1:0]   b_in,
    input  logic [DATA_WIDTH*2-1:0] c_in,
    input  logic                    valid_a_in,
    input  logic                    valid_b_in,
    output logic [DATA_WIDTH-1:0]   a_out,
    output logic [DATA_WIDTH-1:0]   b_out,
    output logic [DATA_WIDTH*2-1:0] c_out,
    output logic                    valid_a_out,
    output logic                    valid_b_out,
    output logic                    valid_c_out
);

logic [DATA_WIDTH*2-1:0] mult_result;
logic [DATA_WIDTH*2-1:0] acc_result;

// Multiply and accumulate
always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        a_out <= 0;
        b_out <= 0;
        c_out <= 0;
        valid_a_out <= 0;
        valid_b_out <= 0;
        valid_c_out <= 0;
        mult_result <= 0;
        acc_result <= 0;
    end else begin
        // Pass through data
        a_out <= a_in;
        b_out <= b_in;
        valid_a_out <= valid_a_in;
        valid_b_out <= valid_b_in;
        
        // Compute multiply-accumulate
        if (valid_a_in && valid_b_in) begin
            mult_result <= $signed(a_in) * $signed(b_in);
            acc_result <= c_in + mult_result;
            c_out <= acc_result;
            valid_c_out <= 1;
        end else begin
            c_out <= c_in;
            valid_c_out <= 0;
        end
    end
end

endmodule

// Simple FIFO implementation
module fifo #(
    parameter DATA_WIDTH = 16,
    parameter DEPTH = 256,
    parameter ADDR_WIDTH = $clog2(DEPTH)
)(
    input  logic                    clk,
    input  logic                    rst_n,
    input  logic                    wr_en,
    input  logic                    rd_en,
    input  logic [DATA_WIDTH-1:0]   wr_data,
    output logic [DATA_WIDTH-1:0]   rd_data,
    output logic                    empty,
    output logic                    full
);

logic [DATA_WIDTH-1:0] mem [DEPTH-1:0];
logic [ADDR_WIDTH:0] wr_ptr, rd_ptr;

assign empty = (wr_ptr == rd_ptr);
assign full = (wr_ptr[ADDR_WIDTH-1:0] == rd_ptr[ADDR_WIDTH-1:0]) && 
              (wr_ptr[ADDR_WIDTH] != rd_ptr[ADDR_WIDTH]);

always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        wr_ptr <= 0;
        rd_ptr <= 0;
        rd_data <= 0;
    end else begin
        if (wr_en && !full) begin
            mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
            wr_ptr <= wr_ptr + 1;
        end
        
        if (rd_en && !empty) begin
            rd_data <= mem[rd_ptr[ADDR_WIDTH-1:0]];
            rd_ptr <= rd_ptr + 1;
        end
    end
end

endmodule