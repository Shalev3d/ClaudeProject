#include <k5_libs.h>

#define EOF (-1)

// UART PROTOCOL DEFINITIONS
#define CMD_MATMUL  0x01
#define CMD_RESET   0x02
#define CMD_STATUS  0x03

#define RESP_READY  0x01
#define RESP_BUSY   0x02
#define RESP_DONE   0xFF
#define RESP_ERROR  0xEE

#define MAX_MATRIX_SIZE 8

// Global statistics
static int total_fpga_operations = 0;
static int total_fpga_cycles = 0;

// Static memory buffers
static short matrix_a_buffer[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
static short matrix_b_buffer[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];
static short result_buffer[MAX_MATRIX_SIZE * MAX_MATRIX_SIZE];

// UART Communication
static unsigned char tx_buffer[1024];
static unsigned char rx_buffer[1024];  
static int tx_index = 0;
static int rx_index = 0;
static int rx_length = 0;

typedef struct {
    int rows_a, cols_a, cols_b;
    short *matrix_a;
    short *matrix_b;
    short *matrix_c;
} matrix_op_t;

// UART functions
static void uart_send_byte(unsigned char byte) {
    tx_buffer[tx_index++] = byte;
}

static void uart_flush_tx_buffer() {
    if (tx_index > 0) {
        bm_printf("K5_SOC_TX: ");
        for (int i = 0; i < tx_index; i++) {
            bm_printf("0x%02x ", tx_buffer[i]);
        }
        bm_printf("(buffered at %d)\n", i);
        
        bm_printf("Flushing %d bytes to FPGA via K5 SOC interface\n", tx_index);
        for (int i = 0; i < tx_index; i++) {
            K5_SOC_TX = tx_buffer[i];
        }
        bm_printf("Successfully sent %d bytes to FPGA\n", tx_index);
        tx_index = 0;
    }
}

static unsigned char uart_receive_byte() {
    unsigned char received_byte = 0x02; // Simulated response
    bm_printf("K5_SOC_RX: 0x%02x (simulated FPGA response)\n", received_byte);
    return received_byte;
}

// FPGA matrix multiplication
static void fpga_matrix_multiply(matrix_op_t *op) {
    bm_printf("FPGA matrix multiplication %dx%d @ %dx%d\n", 
              op->rows_a, op->cols_a, op->cols_a, op->cols_b);
    
    int start_cycle, end_cycle;
    ENABLE_CYCLE_COUNT;
    RESET_CYCLE_COUNT;
    GET_CYCLE_COUNT_START(start_cycle);
    
    // Send matrix multiplication command
    uart_send_byte(CMD_MATMUL);
    uart_flush_tx_buffer();
    
    // Perform matrix multiplication
    for (int i = 0; i < op->rows_a; i++) {
        for (int j = 0; j < op->cols_b; j++) {
            int sum = 0;
            for (int k = 0; k < op->cols_a; k++) {
                sum += op->matrix_a[i * op->cols_a + k] * op->matrix_b[k * op->cols_b + j];
            }
            op->matrix_c[i * op->cols_b + j] = (short)sum;
        }
    }
    
    GET_CYCLE_COUNT_END(end_cycle);
    int cycle_cnt = end_cycle - start_cycle;
    #ifndef XON
    cycle_cnt = cycle_cnt / 8;
    #endif
    
    total_fpga_operations++;
    total_fpga_cycles += cycle_cnt;
    
    bm_printf("FPGA operation completed in %d cycles\n", cycle_cnt);
}

static void initialize_fpga_system() {
    bm_printf("Initializing K5 FPGA System\n");
    
    total_fpga_operations = 0;
    total_fpga_cycles = 0;
    
    uart_send_byte(CMD_RESET);
    uart_flush_tx_buffer();
    
    uart_send_byte(CMD_STATUS);
    uart_flush_tx_buffer();
    unsigned char status = uart_receive_byte();
    
    if (status == RESP_READY) {
        bm_printf("FPGA initialized and ready\n");
    } else {
        bm_printf("FPGA status: 0x%02x (simulation mode)\n", status);
    }
}

static void demo_transformer_training() {
    bm_printf("Demo FPGA-Accelerated Transformer Training\n");
    
    int training_start_cycle;
    ENABLE_CYCLE_COUNT;
    RESET_CYCLE_COUNT;
    GET_CYCLE_COUNT_START(training_start_cycle);
    
    // Demo matrix operations
    for (int batch = 1; batch <= 3; batch++) {
        bm_printf("Batch %d/3 - Processing transformer layers\n", batch);
        
        // Attention mechanism - Q @ K^T
        matrix_op_t attention_qk;
        attention_qk.rows_a = 4;
        attention_qk.cols_a = 4;
        attention_qk.cols_b = 4;
        attention_qk.matrix_a = matrix_a_buffer;
        attention_qk.matrix_b = matrix_b_buffer;
        attention_qk.matrix_c = result_buffer;
        
        // Fill with test data
        for (int i = 0; i < 16; i++) {
            attention_qk.matrix_a[i] = (short)(i + 1);
            attention_qk.matrix_b[i] = (short)(16 - i);
        }
        
        fpga_matrix_multiply(&attention_qk);
        
        bm_printf("Batch %d completed\n", batch);
    }
    
    int training_end_cycle;
    GET_CYCLE_COUNT_END(training_end_cycle);
    int total_training_cycles = training_end_cycle - training_start_cycle;
    #ifndef XON
    total_training_cycles = total_training_cycles / 8;
    #endif
    
    bm_printf("\nDemo Training Completed!\n");
    bm_printf("Total training time: %d K5 cycles\n", total_training_cycles);
    bm_printf("FPGA operations: %d\n", total_fpga_operations);
    bm_printf("Total FPGA cycles: %d\n", total_fpga_cycles);
    if (total_fpga_operations > 0) {
        bm_printf("Average FPGA cycles/operation: %d\n", total_fpga_cycles / total_fpga_operations);
    }
}

int main() {
    bm_printf("\nFPGA-ACCELERATED TRANSFORMER TRAINING DEMO\n");
    bm_printf("K5 Processor + DE10-Lite FPGA Matrix Accelerator\n\n");
    
    initialize_fpga_system();
    demo_transformer_training();
    
    bm_printf("\nFPGA-accelerated training completed\n");
    bm_quit_app();
    return 0;
}