# <img class='emoji_nav' src='emojis/construction.svg' /> OmniOpt2 Architecture Overview

<!-- What is the Architecture of OmniOpt2? -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

This document describes the architecture of the OmniOpt2 system, highlighting its components and workflow modes.

## User Interface

- **User**: The human operator interacting with the system.  
- **[Web GUI](gui)**: A web-based graphical interface that allows users to generate CLI commands easily. This is the primary and only interface for user interaction and command generation.

The [Web GUI](gui) let's the user create the CLI command that drive the OmniOpt2 workflow.

## OmniOpt2 Workflow

OmniOpt2 operates primarily via a CLI (Command Line Interface) command `omniopt ...`. This command controls the entire optimization workflow.

### Local Mode

In Local Mode, OmniOpt2 runs on the user's local system with the following components:

- **OmniOpt2 main script**: The main execution engine responsible for orchestrating jobs.  
- **Jobs (#1 to #n)**: Individual optimization tasks or experiments executed sequentially (parallel execution only in HPC mode)
- **Local Runs Storage**: A local directory (`runs/`) where all results and logs are saved.

The flow is:

- The CLI command triggers the OmniOpt2 Core.  
- The main script manages and distributes jobs
- Each job produces results written back to the local storage.  

### HPC Mode (High Performance Computing)

For larger scale or parallelized workloads, OmniOpt2 supports HPC clusters using `submitit` or `sbatch` job submission.

The HPC setup includes:

- **Login Node**: The initial access point to the cluster where users log in via SSH.  
- **Head Node**: Runs the OmniOpt2 main script that manages job submissions.  
- **Compute Nodes**: Actual machines in the cluster executing jobs (`Job #1`, `Job #2`, ..., `Job #n`).  
- **Runs Storage**: A filesystem accessible by the cluster where results are saved.

The flow is:

- The user logs in to the cluster's Login Node using SSH.  
- The Login Node submits the job batch command to the Head Node, this is done by the OmniOpt2-main-script
- The Head Node runs the OmniOpt2 Core, which submits individual jobs to Compute Nodes.  
- Compute Nodes execute their jobs and return results back to the Head Node.  
- The Head Node writes results to the HPC storage directory.
- (Optional: The results are written to the [Share-Server](tutorials?tutorial=oo_share))

## Visualization

<img style="max-width: 100%;" data-lightsrc="documentation/output_light/architecture.svg" data-darksrc="documentation/output_dark/architecture.svg" />

## Visualization of the interactions of OmniOpt2, Ax, BoTorch and Slurm

<img style="max-width: 100%;" data-lightsrc="documentation/output_light_slurm/slurm.svg" data-darksrc="documentation/output_dark_slurm/slurm.svg" />
