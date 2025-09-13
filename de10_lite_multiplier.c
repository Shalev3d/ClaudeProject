
#include <k5_libs.h>

/****************************** MACROS ******************************/

// NOTICE following must be compliant with the relative used address in the accelerator code, this is NOT automated.

// K5 XBOX FPGA REGISTER DEFINITIONS
#define XBOX_REGS_BASE_ADDR 0x80000000  // K5 FPGA register base address (adjust if needed)

// FPGA Matrix Multiplier Registers
#define FPGA_CMD_REG_IDX      0
#define FPGA_STATUS_REG_IDX   1  
#define FPGA_ROWS_A_REG_IDX   2
#define FPGA_COLS_A_REG_IDX   3
#define FPGA_COLS_B_REG_IDX   4
#define FPGA_DATA_REG_IDX     5   // Data transfer register

// Register macros  
#define FPGA_CMD_REG    ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_CMD_REG_IDX)))
#define FPGA_STATUS_REG ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_STATUS_REG_IDX)))
#define FPGA_ROWS_A_REG ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_ROWS_A_REG_IDX)))
#define FPGA_COLS_A_REG ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_COLS_A_REG_IDX)))
#define FPGA_COLS_B_REG ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_COLS_B_REG_IDX)))  
#define FPGA_DATA_REG   ((volatile unsigned int *) (XBOX_REGS_BASE_ADDR + (4*FPGA_DATA_REG_IDX)))

// Commands
#define CMD_MATMUL  0x01
#define CMD_STATUS  0x03
#define STATUS_READY 0x01
#define STATUS_DONE  0x02 

//---------------------------------------------------------------------------------------------

// K5 FPGA Register Communication Functions

void fpga_send_data(unsigned int data) {
    // Send 32-bit data to FPGA via register interface
    *FPGA_DATA_REG = data;
    bm_printf("FPGA_REG_TX: 0x%08x\n", data);
}

unsigned int fpga_read_data() {
    // Read 32-bit data from FPGA via register interface
    unsigned int data = *FPGA_DATA_REG;
    bm_printf("FPGA_REG_RX: 0x%08x\n", data);
    return data;
}

void fpga_send_command(unsigned int cmd) {
    // Send command to FPGA
    *FPGA_CMD_REG = cmd;
    bm_printf("FPGA_CMD: 0x%08x\n", cmd);
}

unsigned int fpga_get_status() {
    // Get FPGA status
    unsigned int status = *FPGA_STATUS_REG;
    bm_printf("FPGA_STATUS: 0x%08x\n", status);
    return status;
}

void fpga_wait_ready() {
    // Wait for FPGA to be ready
    bm_printf("Waiting for FPGA ready...\n");
    while ((*FPGA_STATUS_REG & STATUS_READY) == 0) {
        // Poll status register
    }
    bm_printf("FPGA is ready\n");
}

void fpga_wait_done() {
    // Wait for FPGA operation to complete  
    bm_printf("Waiting for FPGA completion...\n");
    while ((*FPGA_STATUS_REG & STATUS_DONE) == 0) {
        // Poll status register
    }
    bm_printf("FPGA operation completed\n");
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

// FPGA Accelerated matrix multiplication via K5 register interface

void matrix_multiply_xlr(matrix_config_t *matrix_config_p) {
    bm_printf("Starting FPGA matrix multiplication via K5 registers\n");
    
    // Step 1: Wait for FPGA to be ready
    fpga_wait_ready();
    
    // Step 2: Configure matrix dimensions
    bm_printf("Configuring dimensions: %dx%d * %dx%d\n", 
              matrix_config_p->rows_a, matrix_config_p->cols_a, 
              matrix_config_p->cols_a, matrix_config_p->cols_b);
              
    *FPGA_ROWS_A_REG = matrix_config_p->rows_a;
    *FPGA_COLS_A_REG = matrix_config_p->cols_a;
    *FPGA_COLS_B_REG = matrix_config_p->cols_b;
    
    // Step 3: Send Matrix A data (16-bit elements packed into 32-bit registers)
    bm_printf("Sending Matrix A data (%d elements)\n", matrix_config_p->rows_a * matrix_config_p->cols_a);
    int total_a_elements = matrix_config_p->rows_a * matrix_config_p->cols_a;
    for (int i = 0; i < total_a_elements; i += 2) {
        unsigned int packed_data;
        if (i + 1 < total_a_elements) {
            // Pack two 16-bit elements into one 32-bit register
            packed_data = ((unsigned int)matrix_config_p->matrix_a_addr[i+1] << 16) | 
                         (matrix_config_p->matrix_a_addr[i] & 0xFFFF);
        } else {
            // Only one element left
            packed_data = matrix_config_p->matrix_a_addr[i] & 0xFFFF;
        }
        fpga_send_data(packed_data);
    }
    
    // Step 4: Send Matrix B data
    bm_printf("Sending Matrix B data (%d elements)\n", matrix_config_p->cols_a * matrix_config_p->cols_b);
    int total_b_elements = matrix_config_p->cols_a * matrix_config_p->cols_b;
    for (int i = 0; i < total_b_elements; i += 2) {
        unsigned int packed_data;
        if (i + 1 < total_b_elements) {
            // Pack two 16-bit elements into one 32-bit register
            packed_data = ((unsigned int)matrix_config_p->matrix_b_addr[i+1] << 16) | 
                         (matrix_config_p->matrix_b_addr[i] & 0xFFFF);
        } else {
            // Only one element left
            packed_data = matrix_config_p->matrix_b_addr[i] & 0xFFFF;
        }
        fpga_send_data(packed_data);
    }
    
    // Step 5: Start matrix multiplication
    bm_printf("Starting FPGA matrix multiplication\n");
    fpga_send_command(CMD_MATMUL);
    
    // Step 6: Wait for completion
    fpga_wait_done();
    
    // Step 7: Read result Matrix C data
    bm_printf("Reading result Matrix C data (%d elements)\n", matrix_config_p->rows_a * matrix_config_p->cols_b);
    int total_c_elements = matrix_config_p->rows_a * matrix_config_p->cols_b;
    for (int i = 0; i < total_c_elements; i += 2) {
        unsigned int packed_result = fpga_read_data();
        
        // Unpack two 16-bit results from 32-bit register
        matrix_config_p->matrix_c_addr[i] = (short)(packed_result & 0xFFFF);
        if (i + 1 < total_c_elements) {
            matrix_config_p->matrix_c_addr[i+1] = (short)((packed_result >> 16) & 0xFFFF);
        }
    }
    
    bm_printf("✅ FPGA matrix multiplication completed successfully\n");
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
  
  boolean is_xlr_enabled = FALSE ; // Is Accelerator Enabled , default (can be changed)
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
  bm_sys_call("python3 app_src_dir/check_matrix_result.py");

  bm_quit_app();  // flag to trigger execution termination   
  return 0;
}

//-----------------------------------------------------------------------------------------------