#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <signal.h>

// Funktion zum Behandeln des SIGINT-Signals (Ctrl+C)
void sigint_handler(int sig) {
    printf("\nCtrl+C gedrückt. Beende das Programm.\n");
    exit(0);
}

int main() {
    // Registrieren des Signalhandlers für SIGINT (Ctrl+C)
    signal(SIGINT, sigint_handler);

    // Warnung anzeigen
    printf("WARNUNG: Dieses Programm wird kontinuierlich Speicher allozieren. Drücken Sie Ctrl+C innerhalb von 10 Sekunden, um es zu stoppen.\n");

    // Warten für 10 Sekunden
    sleep(10);

    // Kontinuierlich Speicher allozieren
    while (1) {
        // Speicher dynamisch allozieren, ohne auf NULL zu prüfen
        malloc(1024); // Allocate 1024 bytes
    }

    return 0;
}

