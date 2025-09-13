# Quartus TCL script to program the FPGA
# Run with: quartus_sh -t program.tcl

# Check if programming file exists
set sof_file "output_files/de10_lite_matrix_multiplier.sof"

if {![file exists $sof_file]} {
    puts "Error: Programming file $sof_file not found!"
    puts "Please run compilation first."
    exit 1
}

puts "Programming FPGA with $sof_file..."

# Program the device using USB-Blaster
if {[catch {
    # Auto-detect cable
    set cable_list [get_hardware_names]
    if {[llength $cable_list] == 0} {
        puts "Error: No programming cables found!"
        puts "Please check USB-Blaster connection."
        exit 1
    }
    
    set cable_name [lindex $cable_list 0]
    puts "Using cable: $cable_name"
    
    # Get device list
    set device_list [get_device_names -hardware_name $cable_name]
    if {[llength $device_list] == 0} {
        puts "Error: No devices found on cable!"
        exit 1
    }
    
    set device_name [lindex $device_list 0]
    puts "Programming device: $device_name"
    
    # Program the device
    device_load_waveform -hardware_name $cable_name -device_name $device_name
    device_lock -hardware_name $cable_name -device_name $device_name
    device_virtual_ir_scan -hardware_name $cable_name -device_name $device_name -ir_value 02
    device_virtual_dr_scan -hardware_name $cable_name -device_name $device_name -dr_value [format %08x [file size $sof_file]]
    device_download_sof -hardware_name $cable_name -device_name $device_name -sof $sof_file
    device_unlock -hardware_name $cable_name -device_name $device_name
    
    puts "Programming completed successfully!"
    
} result]} {
    puts "Programming failed: $result"
    
    # Try alternative method using quartus_pgm
    puts "Trying alternative programming method..."
    if {[catch {exec quartus_pgm -c "USB-Blaster" -m jtag -o "p;$sof_file@1"} result]} {
        puts "Alternative programming failed: $result"
        puts "\nTroubleshooting steps:"
        puts "1. Check USB cable connection"
        puts "2. Verify FPGA is powered on" 
        puts "3. Try: quartus_pgm --list to see available cables"
        puts "4. Manual programming: quartus_pgm -c \"USB-Blaster\" -m jtag -o \"p;$sof_file@1\""
        exit 1
    } else {
        puts "Programming completed using quartus_pgm!"
    }
}