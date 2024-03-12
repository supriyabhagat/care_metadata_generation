
# Setup AWS EC2 for Code Execution

## Launch an AWS EC2 Instance
1. Choose Image: Ubuntu Pro Linux
2. Architecture: 64-bit (x86)
3. Instance Type: g4dn.2xlarge
4. Storage: 64 GB
5. Generate a pem key and save it in a directory
6. Network to be decided as per system requirements

## Access EC2 Instance
1. Locate the pem key in your system and revoke access except admin.
2. Open a command prompt as an administrator and set the working directory where the pem key is stored.
3. Go to the EC2 console and connect to the recently created EC2 instance.
4. Use SSH to connect to your EC2 instance. You will need the private key associated with your instance.
   
   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-instance-public-ip
   ```
   
   Paste this command in the command prompt to establish a connection to EC2.


Follow these instructions to set up the project on a Linux system with an NVIDIA GPU.

## GPU Setup

- **Purge Existing NVIDIA Drivers:**
  - `sudo apt-get purge nvidia*`
  - *Removes any existing NVIDIA drivers to avoid conflicts.*

- **Add NVIDIA Graphics Driver Repository:**
  - `sudo add-apt-repository ppa:graphics-drivers/ppa` 
  - `sudo apt update`
  - *Adds the official NVIDIA PPA for the latest drivers.*

- **Install Specific NVIDIA Driver:**
  - `sudo apt install nvidia-driver-535`
  - *Replace 535 with the latest driver version number from the NVIDIA website. Reboot system after installation.*

- **Reboot System:**
  - `sudo reboot`
  - *Reboots your system to apply changes.*

- **Verify Installation:**
  - `nvidia-smi`
  - *Checks and displays details about the GPU and the installed driver version.*

- **Install CUDA Toolkit:**
  - `sudo apt install nvidia-cuda-toolkit` 
  - `nvcc --version`
  - *Installs CUDA Toolkit for GPU acceleration and verifies the installation.*

## Project Setup

- **Update System Packages and Add Python Repository:**
  - `sudo apt update`
  - `sudo apt install software-properties-common`
  - `sudo add-apt-repository ppa:deadsnakes/ppa`
  - `sudo apt install python3`
  - `sudo apt install python3-pip`
  - *Ensures system packages are up to date and installs Python 3.*

## Python Environment Setup

- **Install Virtual Environment Tools:**
  - `pip install virtualenv`
  - `sudo apt install python3-venv`

- **Create and Activate Virtual Environment:**
  - `python3 -m venv venv`
  - `source venv/bin/activate`
  - `pip install --upgrade pip`

- **Transfer Your Python Script from Local Machine to EC2**
   -  Open another command prompt with admin access and set the working directory where the pem key is stored.
   - Use the `scp` command to transfer your Python script from your local machine to the EC2 instance.

   ```bash
   scp -i your-key.pem your-script.py ec2-user@your-ec2-instance-public-ip:/path/to/remote/location
   ```

- **Install Required Packages:**
  - `pip install -r requirements.txt`


## Code Execution
Activate the virtual environment and run the script
- `source venv/bin/activate`
- `python3 main.py`

##  Files 
- utils.py : contains all the functions required for creating vector db, Q/A, model and pipeline creation
- main.py : main code to execute the pipeline
- configs.py: variable/configuration storage
- git_confluence_utils.py : function to fetch data from github and confluence documents
- test_files/preprocessing.py : sample test file for execution
- requirements.txt : all the required libraries with respective versions  



## Notes:

- Always check for the latest versions of software and dependencies for compatibility.
- Replace `535` with the actual latest driver version from the NVIDIA website.
- It's important to verify the installation of each component before proceeding to the next step.

