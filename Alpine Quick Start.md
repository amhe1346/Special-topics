Alpine Quick Start

Alpine is the University of Colorado Boulder Research Computing’s third-generation High-Performance Computing (HPC) cluster. Alpine is a heterogeneous compute cluster currently composed of hardware provided from University of Colorado Boulder, Colorado State University, and Anschutz Medical Campus. Alpine currently offers 489 compute nodes and a total of 32,368 cores. Alpine can be securely accessed anywhere, anytime using Open OnDemand or ssh connectivity to the CURC system.
Alpine Quick-Start

    From a login node, load the slurm/alpine module to access the SLURM job scheduler instance for Alpine:

    $ module load slurm/alpine

    Once the Alpine Slurm job scheduler has been loaded you can submit and start jobs on the Alpine cluster. Consult the requesting resources section and the examples section to learn how to direct your jobs to the appropriate Alpine compute nodes.

    Software can be loaded into the Alpine compute environment via the LMOD module system, which allows users choose software from our pre-installed software stack.

    If you would like to use software that is not within our preinstalled stack, your application must be compiled using acompile. More information about the acompile command can be found under our partitions section.

Hint

    For a list of CURC provided software, see our CURC-Provided Software page.

        Users can also submit a software installation request by following the instructions provided under the FAQ How can I submit a software installation request? dropdown.

    Consult our compiling and linking documentation for more information on compiling software.

Cluster Summary
Nodes

The Alpine cluster is made up of different types of nodes. A general overview of these nodes is as follows:

    CPU nodes: 424 AMD CPU compute nodes

    GPU nodes: a mixture of 41 NVIDIA and AMD GPU nodes

    High-memory nodes: 24 high-memory nodes with 1 TB of memory or more

Alpine is made possible due to its contributors. Below we provide each contributor, along with the number of nodes they have provided:

    University of Colorado Boulder: 375 nodes

    Colorado State University: 77 nodes

    CU Anschutz Medical Campus: 37 nodes

Note

    For specific details on node types and a breakdown of contributions made to Alpine, please see the Hardware Summary section of our Alpine Hardware page.

    All nodes are available to all users via the Slurm Fair Tree Fairshare Algoritm. For additional details on how to check your priority and shares, see the FAQ entitled How can I see my current FairShare priority?

Interconnect

The Alpine cluster has different types of interconnects/fabrics which connect different types of hardware, outlined below:

    CPU nodes: HDR-100 InfiniBand (200Gb inter-node fabric); available on most CPU nodes

    GPU nodes: 2x25 Gb Ethernet +RoCE

    High-memory nodes: A mixture of 2x25 Gb Ethernet +RoCE and 100 Gb Ethernet

    Scratch storage: 25Gb Ethernet +RoCE

Job Scheduling

All jobs on Alpine are run through a queue system using the SLURM job scheduler. Though many HPC workflows are run through batch-type jobs, interactive jobs on compute nodes are allowed; however, these jobs must also be initiated through the scheduler. High-priority jobs move to the top of the queue and are thus guaranteed to start running within a few minutes, unless other high-priority jobs are already queued or running ahead of them. High-priority jobs can run for a maximum wall time of 24 hours. Low-priority jobs have a maximum wall time of 7 days.

More details about how to use SLURM to run jobs can be found in our running applications with jobs documentation.
