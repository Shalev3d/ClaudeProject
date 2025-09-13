#!/bin/bash
# Automated Quartus Prime Lite installation script for Linux
# Supports Ubuntu/Debian and CentOS/RHEL

set -e

echo "🚀 Quartus Prime Lite Installation Script"
echo "========================================="

# Configuration
QUARTUS_VERSION="22.1std.2"
QUARTUS_BUILD="92"
INSTALL_DIR="/opt/intelFPGA_lite"
DOWNLOAD_URL="https://download.altera.com/akdlm/software/acdsinst/${QUARTUS_VERSION}/${QUARTUS_BUILD}/ib_installers"

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$NAME
    VER=$VERSION_ID
elif type lsb_release >/dev/null 2>&1; then
    OS=$(lsb_release -si)
    VER=$(lsb_release -sr)
else
    echo "Cannot detect Linux distribution. Please install manually."
    exit 1
fi

echo "Detected OS: $OS $VER"

# Install prerequisites
echo "📦 Installing prerequisites..."

if [[ "$OS" == *"Ubuntu"* ]] || [[ "$OS" == *"Debian"* ]]; then
    sudo apt-get update
    sudo apt-get install -y \
        wget curl \
        build-essential \
        libc6-dev \
        libncurses5 \
        libxtst6 \
        libxft2 \
        libxext6 \
        unzip \
        default-jre \
        libc6:i386 \
        libncurses5:i386 \
        libstdc++6:i386 \
        libxft2:i386 \
        libxext6:i386

elif [[ "$OS" == *"CentOS"* ]] || [[ "$OS" == *"Red Hat"* ]] || [[ "$OS" == *"Fedora"* ]]; then
    sudo yum update -y
    sudo yum install -y \
        wget curl \
        gcc gcc-c++ \
        glibc-devel \
        ncurses-libs \
        libXtst \
        libXft \
        libXext \
        unzip \
        java-1.8.0-openjdk \
        glibc.i686 \
        ncurses-libs.i686 \
        libstdc++.i686 \
        libXft.i686 \
        libXext.i686

else
    echo "⚠️  Unsupported distribution. You may need to install prerequisites manually."
fi

# Create download directory
mkdir -p ~/quartus_download
cd ~/quartus_download

# Download Quartus (if not already present)
INSTALLER="QuartusLiteSetup-${QUARTUS_VERSION}.${QUARTUS_BUILD}-linux.run"

if [ ! -f "$INSTALLER" ]; then
    echo "⬇️  Downloading Quartus Prime Lite..."
    echo "This is a large file (~3GB), please be patient..."
    wget "${DOWNLOAD_URL}/${INSTALLER}" || {
        echo "❌ Download failed. Please check your internet connection."
        echo "Manual download: ${DOWNLOAD_URL}/${INSTALLER}"
        exit 1
    }
else
    echo "✅ Installer already downloaded"
fi

# Make installer executable
chmod +x "$INSTALLER"

# Create installation directory
sudo mkdir -p "$INSTALL_DIR"
sudo chown $USER:$USER "$INSTALL_DIR"

# Install Quartus
echo "🔧 Installing Quartus Prime Lite..."
echo "This will take 10-20 minutes..."

# Run installer in unattended mode
./"$INSTALLER" \
    --mode unattended \
    --accept_eula 1 \
    --installdir "$INSTALL_DIR" \
    --enable-components quartus,modelsim_ase,devinfo

# Add to PATH
echo "🔗 Configuring environment..."

# Add to bashrc
if ! grep -q "intelFPGA_lite" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Intel FPGA Quartus Prime" >> ~/.bashrc
    echo "export PATH=\"$INSTALL_DIR/${QUARTUS_VERSION}/quartus/bin:\$PATH\"" >> ~/.bashrc
    echo "export PATH=\"$INSTALL_DIR/${QUARTUS_VERSION}/modelsim_ase/bin:\$PATH\"" >> ~/.bashrc
    echo "export QSYS_ROOTDIR=\"$INSTALL_DIR/${QUARTUS_VERSION}/quartus/sopc_builder/bin\"" >> ~/.bashrc
fi

# Source the changes
export PATH="$INSTALL_DIR/${QUARTUS_VERSION}/quartus/bin:$PATH"
export PATH="$INSTALL_DIR/${QUARTUS_VERSION}/modelsim_ase/bin:$PATH"

# Setup USB Blaster permissions
echo "🔌 Configuring USB-Blaster permissions..."

sudo tee /etc/udev/rules.d/51-altera-usb-blaster.rules > /dev/null <<EOF
# Altera USB-Blaster
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6001", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6002", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6003", MODE="0666"

# Altera USB-Blaster II
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6010", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="09fb", ATTR{idProduct}=="6810", MODE="0666"
EOF

sudo udevadm control --reload-rules
sudo udevadm trigger

# Test installation
echo "🧪 Testing installation..."

if command -v quartus_sh >/dev/null 2>&1; then
    echo "✅ Quartus installed successfully!"
    echo "Version: $(quartus_sh --version | head -1)"
else
    echo "❌ Installation may have failed."
    echo "Please check logs and try manual installation."
    exit 1
fi

# Cleanup
echo "🧹 Cleaning up..."
cd ..
rm -rf ~/quartus_download

echo ""
echo "🎉 Installation Complete!"
echo "========================"
echo ""
echo "Quartus Prime Lite is now installed at: $INSTALL_DIR"
echo ""
echo "To use Quartus in new terminal sessions, run:"
echo "  source ~/.bashrc"
echo ""
echo "Or manually add to your PATH:"
echo "  export PATH=\"$INSTALL_DIR/${QUARTUS_VERSION}/quartus/bin:\$PATH\""
echo ""
echo "Next steps:"
echo "1. Connect your DE10-Lite board via USB"
echo "2. Navigate to your FPGA project directory"  
echo "3. Run: make compile"
echo "4. Run: make program"
echo ""
echo "For help: make help"