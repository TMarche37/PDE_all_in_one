#!/bin/bash

# Verifica la presenza del flag -s per modalità silenziosa
silent=false
if [ "$1" == "-s" ]; then
    silent=true
fi

# Funzione per processare una directory che contiene (o deve contenere) la cartella "build"
process_build() {
    local base_dir="$1"

    echo "-------------------------------------------------------------"
    echo -e "\nProcessing directory: $base_dir"

    # Se la directory "build" non esiste ma esiste "src", la crea
    if [ ! -d "$base_dir/build" ]; then
        if [ -d "$base_dir/src" ]; then
            echo "Non trovata la cartella 'build' in $base_dir, ma è presente'src\n'. Creazione di 'build'..."
            mkdir "$base_dir/build" || { echo "Impossibile creare la cartella build in $base_dir"; exit 1; }
        else
            echo "Nella directory $base_dir non esiste una cartella 'build' (e nemmeno 'scr')."
            return
        fi
    fi

    # Entra nella directory build
    cd "$base_dir/build" || { echo "Impossibile entrare in $base_dir/build"; exit 1; }

    # Rimuove il contenuto della cartella build
    if [ ! $silent ]; then
	echo "Emptying $base_dir/build"
    fi
	rm -r * > /dev/null 2>&1

    # Esegue cmake e make
    if $silent; then
        cmake .. > /dev/null 2>&1 || { echo "CMake fallito in $base_dir/build"; exit 1; }
        make > /dev/null 2>&1 || { echo "Make fallito in $base_dir/build"; exit 1; }
    else
        cmake .. || { echo "CMake fallito in $base_dir/build"; exit 1; }
        make || { echo "Make fallito in $base_dir/build"; exit 1; }
    fi

    # Esegue l'eseguibile "main" se presente
    if [ -f "./main" ]; then
        echo "Esecuzione di ./main in $base_dir/build..."
        ./main || { echo "Esecuzione di 'main' fallita in $base_dir/build"; exit 1; }
    else
        echo "Eseguibile 'main' non trovato in $base_dir/build"
    fi

    echo "-------------------------------------------------------------"
    echo -e "\n\n"

    # Torna alla directory di partenza
    cd - > /dev/null 2>&1
}

# Itera su ogni sottocartella (livello 1)
for subdir in */; do
    # Se esiste una cartella "src" all'interno della sottocartella, processa direttamente "build" o "scr"
    if [ -d "$subdir/src" ]; then
        if [ -d "$subdir/build" ] || [ -d "$subdir/src" ]; then
            process_build "$subdir"
        else
            echo "Nella directory $subdir non esiste una cartella 'build' né 'rc'."
        fi
    else
        echo "Non trovato 'src' in $subdir, cerco nelle sottocartelle..."
        # Se non esiste "src", considera la directory come contenitore di ulteriori sottocartelle
        for subsub in "$subdir"*/; do
            if [ -d "$subsub/build" ] || [ -d "$subsub/src" ]; then
                process_build "$subsub"
            else
                echo "Nella directory $subsub non esiste una cartella 'build' né 'src'."
            fi
        done
    fi
done
