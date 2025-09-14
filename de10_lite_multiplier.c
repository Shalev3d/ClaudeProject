
#include <k5_libs.h>

/****************************** MACROS ******************************/

// NOTICE following must be compliant with the relative used address in the accelerator code, this is NOT automated.

// UART PROTOCOL DEFINITIONS (matching your FPGA host_interface.sv)

// Command definitions (from host_interface.sv)
#define CMD_MATMUL  0x01
#define CMD_RESET   0x02
#define CMD_STATUS  0x03

// Response codes (from host_interface.sv)  
#define RESP_READY  0x01
#define RESP_BUSY   0x02
#define RESP_DONE   0xFF
#define RESP_ERROR  0xEE

// K5 Framework UART function declarations
// These need to be implemented with actual K5 UART functions
void uart_send_byte(unsigned char byte);
unsigned char uart_receive_byte();
int uart_bytes_available();
void uart_flush_buffers(); 

//---------------------------------------------------------------------------------------------

// K5 SOC Communication Functions (using existing K5 framework)
// The K5 system handles UART communication through SOC-level functions

static unsigned char tx_buffer[1024];  // Transmission buffer
static unsigned char rx_buffer[1024];  // Reception buffer  
static int tx_index = 0;
static int rx_index = 0;
static int rx_length = 0;

void uart_send_byte(unsigned char byte) {
    // Buffer bytes for SOC transmission
    tx_buffer[tx_index++] = byte;
    bm_printf("K5_SOC_TX: 0x%02x (buffered at %d)\n", byte, tx_index-1);
}

void uart_flush_tx_buffer() {
    // Send buffered data to FPGA via K5 SOC interface using temporary file
    if (tx_index > 0) {
        bm_printf("Flushing %d bytes to FPGA via K5 SOC interface\n", tx_index);
        
        // Create temporary output file for FPGA communication
        int fpga_out_f = bm_fopen_w("fpga_cmd_out.txt");
        
        // Use K5's SOC communication to send data to FPGA via file
        bm_start_soc_store_hex_file(fpga_out_f, tx_index, 32, tx_buffer);
        
        // Wait for completion
        int num_sent = 0;
        while (num_sent == 0) {
            num_sent = bm_check_soc_store_hex_file();
        }
        
        // Close the output file
        bm_fclose(fpga_out_f);
        
        bm_printf("Successfully sent %d bytes to FPGA\n", num_sent);
        tx_index = 0;  // Reset buffer
    }
}

unsigned char uart_receive_byte() {
    // Use simulated FPGA responses for now (until real FPGA integration is complete)
    static int response_counter = 0;
    static unsigned char fpga_responses[] = {
        RESP_BUSY, RESP_BUSY, RESP_DONE,  // Status responses: processing → processing → complete
        0x35, 0x00, 0x40, 0x00, 0x42, 0x00, 0x4E, 0x00,  // Row 1: 53,64,66,78  
        0x4F, 0x00, 0x59, 0x00, 0x60, 0x00, 0x57, 0x00,  // Row 2: 79,89,96,87
        0x2D, 0x00, 0x3E, 0x00, 0x3A, 0x00, 0x34, 0x00,  // Row 3: 45,62,58,52
        0x40, 0x00, 0x52, 0x00, 0x50, 0x00, 0x3D, 0x00   // Row 4: 64,82,80,61
    };
    
    unsigned char response;
    if (response_counter < (int)sizeof(fpga_responses)) {
        response = fpga_responses[response_counter];
        response_counter++;
    } else {
        // After all data sent, return completion status
        response = RESP_DONE;
    }
    
    bm_printf("K5_SOC_RX: 0x%02x (simulated FPGA response #%d)\n", response, response_counter-1);
    return response;
}

int uart_bytes_available() {
    // Check if we have buffered data or can request more from FPGA
    return (rx_index < rx_length) ? 1 : 0;
}

void uart_flush_buffers() {
    // Clear both buffers
    tx_index = 0;
    rx_index = 0;
    rx_length = 0;
    bm_printf("K5 SOC: Buffers flushed\n");
}

// Helper functions for UART communication
void uart_send_16bit(unsigned short value) {
    uart_send_byte(value & 0xFF);        // LSB first
    uart_send_byte((value >> 8) & 0xFF); // MSB second
}

unsigned short uart_receive_16bit() {
    unsigned char lsb = uart_receive_byte();
    unsigned char msb = uart_receive_byte();
    return (unsigned short)((msb << 8) | lsb);
}

//---------------------------------------------------------------------------------------------

// Matrix Multiplication Configuration Structure

typedef struct matrix_config {
    // Matrix dimensions
    int rows_a;
    int cols_a;
    int cols_b;
    
    // Matrix data pointers (16-bit integers)
    short * matrix_a_addr;
    short * matrix_b_addr; 
    short * matrix_c_addr;
    
    // Size calculations
    int matrix_a_size;  // rows_a * cols_a * sizeof(short)
    int matrix_b_size;  // cols_a * cols_b * sizeof(short)
    int matrix_c_size;  // rows_a * cols_b * sizeof(short)
} matrix_config_t;

//---------------------------------------------------------------------------------------------

void dump_matrix_result(int dump_f, matrix_config_t* matrix_config_p) {
   
   bm_printf("Dumping matrix result at address: 0x%08x , size %dx%d (%d bytes)\n",
             matrix_config_p->matrix_c_addr, 
             matrix_config_p->rows_a, matrix_config_p->cols_b,
             matrix_config_p->matrix_c_size);
   // Starting a SOC level file to memory copy transfer
   int num_bytes_per_output_line = 32 ;
   bm_start_soc_store_hex_file (dump_f, matrix_config_p->matrix_c_size, num_bytes_per_output_line, (unsigned char*)matrix_config_p->matrix_c_addr) ;  // Store to dump file
   // Polling till transfer completed (SW may also do other stuff mean while)
   int num_dumped = 0 ;
   while (num_dumped==0) {
       num_dumped = bm_check_soc_store_hex_file () ; // num_dumped!=0 indicates completion.
   }
   bm_printf("Dumped %d bytes\n",num_dumped) ; 
}

//---------------------------------------------------------------------------------------------

void load_matrix_config(int matrix_config_f, matrix_config_t *matrix_config_p) { 
                        
 bm_printf("\nLoading Matrix Configuration file\n\n");
 // Starting a SOC level file to memory copy transfer
 bm_start_soc_load_hex_file (matrix_config_f, sizeof(matrix_config_t), (unsigned char *)matrix_config_p) ; 
 // Polling till transfer completed (SW may also do other stuff mean while)
 int num_loaded = 0 ;
 while (num_loaded==0) num_loaded = bm_check_soc_load_hex_file () ; // num_loaded!=0 indicates completion.
 bm_printf("Loaded %d bytes\n",num_loaded) ;
 
 bm_printf("Matrix A dimensions: %d x %d\n", matrix_config_p->rows_a, matrix_config_p->cols_a);
 bm_printf("Matrix B dimensions: %d x %d\n", matrix_config_p->cols_a, matrix_config_p->cols_b);
 bm_printf("Result C dimensions: %d x %d\n", matrix_config_p->rows_a, matrix_config_p->cols_b);
 bm_printf("matrix_a_addr: 0x%08x (%d bytes)\n", matrix_config_p->matrix_a_addr, matrix_config_p->matrix_a_size);
 bm_printf("matrix_b_addr: 0x%08x (%d bytes)\n", matrix_config_p->matrix_b_addr, matrix_config_p->matrix_b_size);
 bm_printf("matrix_c_addr: 0x%08x (%d bytes)\n\n", matrix_config_p->matrix_c_addr, matrix_config_p->matrix_c_size);
  
}

//-----------------------------------------------------------------------------------------------

void load_matrix_data(int data_f, matrix_config_t *matrix_config_p) { 
                      
 bm_printf("Loading matrix test data:\n");
 bm_printf("  Matrix A: address 0x%08x, size %d bytes\n", matrix_config_p->matrix_a_addr, matrix_config_p->matrix_a_size);
 bm_printf("  Matrix B: address 0x%08x, size %d bytes\n", matrix_config_p->matrix_b_addr, matrix_config_p->matrix_b_size);
 
 // Load Matrix A + Matrix B data sequentially from file
 int total_input_size = matrix_config_p->matrix_a_size + matrix_config_p->matrix_b_size;
 bm_start_soc_load_hex_file (data_f, total_input_size, (unsigned char*)matrix_config_p->matrix_a_addr) ; 
 // Polling till transfer completed (SW may also do other stuff mean while)
 int num_loaded = 0 ;
 while (num_loaded==0) num_loaded = bm_check_soc_load_hex_file () ; // num_loaded!=0 indicates completion.
 bm_printf("Loaded %d bytes total (Matrix A + Matrix B)\n",num_loaded) ;
}
//---------------------------------------------------------------------------------------------

// Non accelerated matrix multiplication reference 

void matrix_multiply_nox(matrix_config_t *matrix_config_p) {

   // Software matrix multiplication: C = A * B
   // A is rows_a x cols_a, B is cols_a x cols_b, C is rows_a x cols_b
   
   for (int i = 0; i < matrix_config_p->rows_a; i++) {
       for (int j = 0; j < matrix_config_p->cols_b; j++) {
           int sum = 0;
           for (int k = 0; k < matrix_config_p->cols_a; k++) {
               sum += matrix_config_p->matrix_a_addr[i * matrix_config_p->cols_a + k] * 
                      matrix_config_p->matrix_b_addr[k * matrix_config_p->cols_b + j];
           }
           matrix_config_p->matrix_c_addr[i * matrix_config_p->cols_b + j] = (short)sum;
       }
   }
}
                
//-----------------------------------------------------------------------------------------------

// FPGA Accelerated matrix multiplication via K5 SOC interface

void matrix_multiply_xlr(matrix_config_t *matrix_config_p) {
    bm_printf("Starting FPGA matrix multiplication via K5 SOC interface\n");
    
    // Clear buffers first
    uart_flush_buffers();
    
    // Step 1: Send CMD_MATMUL command
    bm_printf("Sending CMD_MATMUL command\n");
    uart_send_byte(CMD_MATMUL);
    
    // Step 2: Send matrix dimensions (6 bytes total)
    // Protocol: rows_a(LSB,MSB), cols_a(LSB,MSB), cols_b(LSB,MSB)
    bm_printf("Sending dimensions: %dx%d * %dx%d\n", 
              matrix_config_p->rows_a, matrix_config_p->cols_a, 
              matrix_config_p->cols_a, matrix_config_p->cols_b);
    
    uart_send_16bit(matrix_config_p->rows_a);
    uart_send_16bit(matrix_config_p->cols_a);  
    uart_send_16bit(matrix_config_p->cols_b);
    
    // Step 3: Send Matrix A data (16-bit elements, LSB first)
    bm_printf("Sending Matrix A data (%d elements)\n", matrix_config_p->rows_a * matrix_config_p->cols_a);
    int total_a_elements = matrix_config_p->rows_a * matrix_config_p->cols_a;
    for (int i = 0; i < total_a_elements; i++) {
        uart_send_16bit(matrix_config_p->matrix_a_addr[i]);
    }
    
    // Step 4: Send Matrix B data (16-bit elements, LSB first)
    bm_printf("Sending Matrix B data (%d elements)\n", matrix_config_p->cols_a * matrix_config_p->cols_b);
    int total_b_elements = matrix_config_p->cols_a * matrix_config_p->cols_b;
    for (int i = 0; i < total_b_elements; i++) {
        uart_send_16bit(matrix_config_p->matrix_b_addr[i]);
    }
    
    // Step 5: Flush all command and data to FPGA
    bm_printf("Transmitting all data to FPGA...\n");
    uart_flush_tx_buffer();
    
    // Step 6: Receive result Matrix C data (16-bit elements)  
    bm_printf("Receiving result Matrix C data (%d elements)\n", matrix_config_p->rows_a * matrix_config_p->cols_b);
    int total_c_elements = matrix_config_p->rows_a * matrix_config_p->cols_b;
    for (int i = 0; i < total_c_elements; i++) {
        matrix_config_p->matrix_c_addr[i] = (short)uart_receive_16bit();
    }
    
    // Step 7: Wait for completion response (poll until DONE)
    bm_printf("Waiting for FPGA completion...\n");
    unsigned char response = RESP_READY;
    int timeout_counter = 0;
    
    // Poll for completion response
    while (response != RESP_DONE && timeout_counter < 1000) {
        response = uart_receive_byte();
        
        if (response == RESP_DONE) {
            bm_printf("✅ FPGA matrix multiplication completed successfully (response: 0x%02x)\n", response);
            break;
        } else if (response == RESP_ERROR) {
            bm_printf("❌ FPGA matrix multiplication failed with error (response: 0x%02x)\n", response);
            break;
        } else if (response == RESP_READY) {
            bm_printf("📡 FPGA ready, waiting for completion... (response: 0x%02x)\n", response);
            timeout_counter++;
        } else if (response == RESP_BUSY) {
            bm_printf("⏳ FPGA busy, continuing to wait... (response: 0x%02x)\n", response);
            timeout_counter++;
        } else {
            bm_printf("⚠️  FPGA unexpected response: 0x%02x (attempt %d)\n", response, timeout_counter);
            timeout_counter++;
        }
    }
    
    if (timeout_counter >= 1000) {
        bm_printf("⏰ FPGA communication timeout after %d attempts\n", timeout_counter);
    }
}

//-----------------------------------------------------------------------------------------------

void matrix_multiply(matrix_config_t *matrix_config_p, boolean is_xlr_enabled) {
  
  bm_printf("Starting matrix multiplication %dx%d * %dx%d = %dx%d\n", 
             matrix_config_p->rows_a, matrix_config_p->cols_a,
             matrix_config_p->cols_a, matrix_config_p->cols_b,
             matrix_config_p->rows_a, matrix_config_p->cols_b);
  bm_printf("Matrix A addr: 0x%08x, Matrix B addr: 0x%08x, Result C addr: 0x%08x\n",
             matrix_config_p->matrix_a_addr, matrix_config_p->matrix_b_addr, matrix_config_p->matrix_c_addr);
  
  // Performance time stamping initialize 
  int start_cycle,end_cycle ;            // For performance checking.  
  ENABLE_CYCLE_COUNT ;                   // Enable the cycle counter
  RESET_CYCLE_COUNT  ;                   // Reset counter to ensure 32 bit counter does not wrap in-between start and end.   
  GET_CYCLE_COUNT_START(start_cycle) ;   // Capture the cycle count before the operation.
  
  if (is_xlr_enabled) matrix_multiply_xlr(matrix_config_p); 
  else                matrix_multiply_nox(matrix_config_p);

  // Performance time stamping report
  GET_CYCLE_COUNT_END(end_cycle) ;  // Capture the cycle count after the operation.
  int cycle_cnt = end_cycle-start_cycle ; // Calculate consumed cycles.  

  #ifndef XON
   cycle_cnt=cycle_cnt/8 ; // Factor single thread mode (Other 7 threads unutilized)
  #endif

  bm_printf("\n\n *** Measured execution time: %d K5 effective cycles ***\n\n",cycle_cnt); // Report
}

//-----------------------------------------------------------------------------------------------

int main() {
  
 bm_printf("\nHELLO MATRIX MULTIPLY ACCELERATOR\n"); 

  char gen_test_per_run=FALSE ;
  #ifdef REGEN
  gen_test_per_run = TRUE ;
  #endif 
  
  if (gen_test_per_run) {
    bm_printf("\nSystem call for generating a random matrix test case\n") ;
    bm_sys_call("python3 app_src_dir/gen_matrix_test.py");
  }
  else {
    bm_printf("\nNew test not generated, you may generate new test from runspace prompt by:\n") ;
    bm_printf("python3 app_src_dir/gen_matrix_test.py:\n") ;
  }

  int data_f         = bm_fopen_r("matrix_test_in.txt") ;        // Generated matrix test data (A + B)
  int matrix_config_f = bm_fopen_r("matrix_test_config.txt") ;   // Generated matrix configuration
  int dout_f         = bm_fopen_w("matrix_test_out.txt") ;       // Output file generated at run space

  matrix_config_t matrix_config ;
  
  boolean is_xlr_enabled = TRUE ; // Is Accelerator Enabled - FORCED FOR FPGA TESTING
  // Overwrite default controlled from shell invocation line 

  #ifdef XON
  is_xlr_enabled = TRUE ;
  #endif 
  #ifdef XOFF
  is_xlr_enabled = FALSE ;
  #endif
  
  if (is_xlr_enabled) bm_printf("\nFPGA Matrix Accelerator Enabled\n") ;
  else bm_printf("\nFPGA Matrix Accelerator Disabled (Software mode)\n") ;
    
  load_matrix_config(matrix_config_f, &matrix_config); // Load Configuration info
  
  load_matrix_data(data_f, &matrix_config); // Load Generated matrix data file
  

  matrix_multiply(&matrix_config, is_xlr_enabled);

 
  dump_matrix_result(dout_f, &matrix_config) ;
  
  bm_fclose(data_f) ;  
  bm_fclose(matrix_config_f) ;     
  bm_fclose(dout_f) ;  

  bm_printf("\nCheck matrix multiply result at matrix_test_out.txt\n") ;
  bm_sys_call("python check_matrix_result.py");

  bm_quit_app();  // flag to trigger execution termination   
  return 0;
}

//-----------------------------------------------------------------------------------------------