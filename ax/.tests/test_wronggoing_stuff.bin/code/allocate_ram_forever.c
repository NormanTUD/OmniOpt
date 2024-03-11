#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

void sigint_handler(int sig) {
    printf("\nCtrl+C pressed. Exiting.\n");
    exit(0);
}

int main() {
    signal(SIGINT, sigint_handler);

    printf("WARNING: This program allocates memory for ever. It may freeze your computer. Press CTRL+C in the next 10 seconds to cancel it.\n");

    sleep(10);

    int allocated_memory = 0;

    int allocate_per_step = 10 * 1048576; // 1 MB = 1048576 Bytes

    while (1) {
	    malloc(allocate_per_step);
	    allocated_memory += allocate_per_step;
	    printf("Allocated %d MB (step %d)\n", allocated_memory / 1048576, allocated_memory / allocate_per_step);
    }

    return 0;
}

