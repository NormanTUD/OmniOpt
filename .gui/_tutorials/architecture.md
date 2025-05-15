# OmniOpt2 Architecture Overview

<!-- What is the Architecture of OmniOpt2? -->

<!-- Category: Preparations, Basics and Setup -->

<div id="toc"></div>

This document describes the architecture of the OmniOpt2 system, highlighting its components and workflow modes.

## 1. User Interface

- **User**: The human operator interacting with the system.  
- **Web GUI**: A web-based graphical interface that allows users to generate CLI commands easily. This is the primary and only interface for user interaction and command generation.

The Web GUI translates user actions into CLI commands that drive the core OmniOpt2 workflow.

## 2. OmniOpt2 Workflow

The core of OmniOpt2 operates primarily via a CLI (Command Line Interface) command `omniopt ...`. This command controls the entire optimization workflow.

### 2.1 Local Mode

In Local Mode, OmniOpt2 runs on the user's local system with the following components:

- **OmniOpt2 Core (main script)**: The main execution engine responsible for orchestrating jobs.  
- **Jobs (#1 to #n)**: Individual optimization tasks or experiments executed sequentially or in parallel, as determined by the user.  
- **Local Runs Storage**: A local directory (`runs/`) where all results and logs are saved.

The flow is:

1. The CLI command triggers the OmniOpt2 Core.  
2. The Core manages and distributes jobs (`Job #1`, `Job #2`, ..., `Job #n`).  
3. Each job produces results written back to the local storage.  

### 2.2 HPC Mode (High Performance Computing)

For larger scale or parallelized workloads, OmniOpt2 supports HPC clusters using `submitit` or `sbatch` job submission.

The HPC setup includes:

- **Login Node**: The initial access point to the cluster where users log in via SSH.  
- **Head Node**: Runs the OmniOpt2 Core script that manages job submissions.  
- **Compute Nodes**: Actual machines in the cluster executing jobs (`Job #1`, `Job #2`, ..., `Job #n`).  
- **Runs Storage**: A filesystem accessible by the cluster where results are saved.

The flow is:

1. The user logs in to the cluster's Login Node using SSH.  
2. The Login Node submits the job batch command (`sbatch`) to the Head Node.  
3. The Head Node runs the OmniOpt2 Core, which submits individual jobs to Compute Nodes.  
4. Compute Nodes execute their jobs and return results back to the Head Node.  
5. The Head Node writes results to the HPC storage directory.

## Visualization

<img src="documentation/architektur.svg" />
